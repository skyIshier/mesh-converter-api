import struct
import os
import sys
import re
import ctypes
import io
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    import lz4.block
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False
    try:
        from ctypes import CDLL
        lz4 = CDLL('liblz4.so.1')
        LZ4_SO_AVAILABLE = True
    except:
        LZ4_SO_AVAILABLE = False

VERTEX_STRIDE = 16
UV_STRIDE = 16
INDEX_STRIDE_32 = 4
INDEX_STRIDE_16 = 2

HEADER_VERSION_MAP = {
    b'\x17\x00\x00\x00': 23, b'\x18\x00\x00\x00': 24,
    b'\x19\x00\x00\x00': 25, b'\x1a\x00\x00\x00': 26,
    b'\x1b\x00\x00\x00': 27, b'\x1c\x00\x00\x00': 28,
    b'\x1d\x00\x00\x00': 29, b'\x1e\x00\x00\x00': 30,
    b'\x1f\x00\x00\x00': 31, b'\x20\x00\x00\x00': 32,
}


class Reader:
    __slots__ = ("data", "ofs", "size")
    def __init__(self, data: bytes):
        self.data = data; self.ofs = 0; self.size = len(data)
    def tell(self) -> int:
        return self.ofs
    def seek(self, pos: int, whence: int = 0):
        if whence == 0: self.ofs = pos
        elif whence == 1: self.ofs += pos
        elif whence == 2: self.ofs = self.size + pos
        if self.ofs < 0 or self.ofs > self.size:
            raise ValueError(f"Seek out of range: {self.ofs}/{self.size}")
    def read_bytes(self, n: int) -> bytes:
        if self.ofs + n > self.size:
            raise ValueError(f"Read out of range: {self.ofs}+{n} > {self.size}")
        b = self.data[self.ofs:self.ofs+n]; self.ofs += n; return b
    def read_u8(self) -> int: return self._unpack("<B", 1)[0]
    def read_u16(self) -> int: return self._unpack("<H", 2)[0]
    def read_u32(self) -> int: return self._unpack("<I", 4)[0]
    def read_fmt(self, fmt: str):
        sz = struct.calcsize(fmt); return self._unpack(fmt, sz)
    def _unpack(self, fmt, sz):
        if self.ofs + sz > self.size:
            raise ValueError(f"Unpack out of range: {self.ofs}+{sz} > {self.size}")
        out = struct.unpack_from(fmt, self.data, self.ofs); self.ofs += sz; return out


def lz4_block_decompress(src: bytes, uncompressed_size: Optional[int] = None) -> bytes:
    try:
        import lz4.block
        if uncompressed_size is None: return lz4.block.decompress(src)
        return lz4.block.decompress(src, uncompressed_size=uncompressed_size)
    except Exception: pass
    i = 0; out = bytearray(); src_len = len(src)
    def read_len(base):
        nonlocal i
        ln = base
        if ln == 15:
            while True:
                if i >= src_len: raise ValueError("LZ4: truncated")
                s = src[i]; i += 1; ln += s
                if s != 255: break
        return ln
    while i < src_len:
        token = src[i]; i += 1
        lit_len = read_len(token >> 4)
        if i + lit_len > src_len: raise ValueError("LZ4: literal OOB")
        out += src[i:i+lit_len]; i += lit_len
        if i >= src_len: break
        if i + 2 > src_len: raise ValueError("LZ4: missing offset")
        offset = src[i] | (src[i+1] << 8); i += 2
        if offset == 0: raise ValueError("LZ4: offset=0")
        match_len = read_len(token & 0x0F) + 4
        start = len(out) - offset
        if start < 0: raise ValueError("LZ4: offset beyond buf")
        for _ in range(match_len): out.append(out[start]); start += 1
    if uncompressed_size is not None and len(out) != uncompressed_size:
        raise ValueError(f"LZ4: size mismatch {len(out)} vs {uncompressed_size}")
    return bytes(out)


def lz4_block_compress(src: bytes) -> bytes:
    try:
        import lz4.block; return lz4.block.compress(src, store_size=False)
    except ImportError: pass
    try:
        from ctypes import CDLL; _lz4 = CDLL('liblz4.so.1')
        mx = _lz4.LZ4_compressBound(len(src))
        dest = ctypes.create_string_buffer(mx)
        ret = _lz4.LZ4_compress_default(src, dest, len(src), mx)
        if ret <= 0: raise IOError("LZ4 compress failed")
        return dest.raw[:ret]
    except: pass
    raise RuntimeError("没有可用的LZ4压缩方法，请安装lz4库 (pip install lz4)")


def do_lz4_decompress(src, uncompressed_size):
    if LZ4_AVAILABLE:
        return lz4.block.decompress(src, uncompressed_size=uncompressed_size)
    elif LZ4_SO_AVAILABLE:
        dest = ctypes.create_string_buffer(uncompressed_size)
        ret = lz4.LZ4_decompress_safe(src, dest, len(src), uncompressed_size)
        if ret <= 0: raise IOError('LZ4解压失败')
        return dest.raw
    else:
        return lz4_block_decompress(src, uncompressed_size)


def hex_string_to_bytes(hex_str):
    hex_str = re.sub(r'\s+', '', hex_str)
    if not hex_str: return b''
    try: return bytes.fromhex(hex_str)
    except ValueError as e: print(f"16进制字符串格式错误: {e}"); return b''


def find_first_01(data, start_offset=0):
    for i in range(start_offset, len(data)):
        if data[i] == 0x01: return i
    return None


def create_output_folder(filepath, base_name, version):
    base_dir = os.path.dirname(os.path.abspath(filepath))
    obj_dir = os.path.join(base_dir, "obj"); os.makedirs(obj_dir, exist_ok=True)
    out_folder = os.path.join(obj_dir, f"{base_name}_{version}")
    idx = 1; orig = out_folder
    while os.path.exists(out_folder): out_folder = f"{orig}_{idx}"; idx += 1
    os.makedirs(out_folder); return out_folder


def save_results(out_folder, base_name, raw_vertices, raw_uv, raw_indices,
                 vertex_buffer, uv_buffer, triangles,
                 extra_info=None, is_special=False, extra_gap=0, decompressed_raw=None):
    with open(os.path.join(out_folder, f"{base_name}_vertices.bin"), 'wb') as f: f.write(raw_vertices)
    with open(os.path.join(out_folder, f"{base_name}_uvs.bin"), 'wb') as f: f.write(raw_uv)
    with open(os.path.join(out_folder, f"{base_name}_indices.bin"), 'wb') as f: f.write(raw_indices)
    if decompressed_raw is not None:
        with open(os.path.join(out_folder, f"{base_name}_decompressed.bin"), 'wb') as f: f.write(decompressed_raw)
    txt_path = os.path.join(out_folder, f"{base_name}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"源文件: {extra_info.get('filepath','')}\n")
        f.write(f"文件大小: {extra_info.get('file_size',0)} 字节\n")
        if 'decompressed_size' in extra_info: f.write(f"解压后大小: {extra_info['decompressed_size']} 字节\n")
        f.write(f"顶点数: {extra_info.get('vertex_count',0)}\n")
        f.write(f"实际顶点数: {len(vertex_buffer)}\n")
        f.write(f"顶点与UV间隔: {extra_info.get('gap',0)} 字节\n")
        if is_special: f.write(f"特殊文件额外间隙: {extra_gap} 字节\n")
        f.write(f"索引个数: {extra_info.get('index_count',0)}\n")
        f.write(f"实际三角形数: {len(triangles)}\n\n")
        if 'bones' in extra_info and extra_info['bones']:
            f.write("骨骼信息:\n")
            for b in extra_info['bones']: f.write(f"  {b}\n")
            f.write("\n")
        f.write("顶点列表:\n")
        for i, v in enumerate(vertex_buffer): f.write(f"v{i}: {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\nUV列表:\n")
        for i, uv in enumerate(uv_buffer): f.write(f"uv{i}: {uv[0]:.6f} {uv[1]:.6f}\n")
        f.write("\n三角形列表:\n")
        for i, tri in enumerate(triangles): f.write(f"f{i}: {tri[0]} {tri[1]} {tri[2]}\n")
    print(f"报告: {txt_path}")
    obj_path = os.path.join(out_folder, f"{base_name}.obj")
    with open(obj_path, 'w') as f:
        f.write(f"# OBJ from {extra_info.get('filename', base_name)}\n")
        for v in vertex_buffer: f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for uv in uv_buffer: f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        for tri in triangles: f.write(f"f {tri[0]+1}/{tri[0]+1} {tri[1]+1}/{tri[1]+1} {tri[2]+1}/{tri[2]+1}\n")
    print(f"OBJ: {obj_path}")
    try:
        import bpy
        mesh = bpy.data.meshes.new(base_name); mesh.from_pydata(vertex_buffer, [], triangles); mesh.update()
        if uv_buffer:
            uvl = mesh.uv_layers.new(); ud = []
            for loop in mesh.loops: ud.extend(uv_buffer[loop.vertex_index])
            uvl.data.foreach_set('uv', ud)
        obj = bpy.data.objects.new(base_name, mesh); bpy.context.scene.collection.objects.link(obj)
    except: pass
    print(f"输出: {out_folder}")


# ======================= 解析函数 =======================

def process_header_17(data, filepath, filename, version, is_batch, return_data_only=False):
    name_no_ext = os.path.splitext(filename)[0]; file_size = len(data)
    if not return_data_only and "StripAnim" not in filename:
        fn_lower = filename.lower()
        if (("anim" in fn_lower) or ("anc" in fn_lower and not fn_lower.startswith("anc"))) and "衣服" not in filename:
            if is_batch: print(f"跳过: {filename}"); return None
            else: manual_process(); return None
    if not return_data_only and "StripAnim" in filename:
        vip = 0x4061; iip = 0x4065; vs = 0x408D
        if vip+4 > file_size: return None
        vc = struct.unpack('<I', data[vip:vip+4])[0]; vb = vc*16
        if vs+vb > file_size: return None
        raw_v = data[vs:vs+vb]
        vbuf = []
        for i in range(0, vb, 16):
            c = raw_v[i:i+16]
            if len(c)>=12:
                try: x,y,z = struct.unpack('<fff', c[:12]); vbuf.append((x,y,z))
                except: pass
        gap = vb//4; ns = vs+vb; rn = data[ns:ns+gap]
        us = ns+gap; ue = us+vb
        if ue > file_size: ue = file_size
        raw_uv = data[us:ue]
        ubuf = []
        for i in range(0, len(raw_uv), 16):
            c = raw_uv[i:i+16]
            if len(c)>=8:
                try: u,v = struct.unpack('<ff', c[:8]); ubuf.append((u,v))
                except: ubuf.append((0.0,0.0))
        while len(ubuf) < len(vbuf): ubuf.append((0.0,0.0))
        if len(ubuf) > len(vbuf): ubuf = ubuf[:len(vbuf)]
        eg = vc*8
        if iip+4 > file_size: return None
        ic = struct.unpack('<I', data[iip:iip+4])[0]; ib = ic*4
        idx_s = ue+eg; idx_e = idx_s+ib
        if idx_e > file_size: idx_e = file_size
        if idx_s >= file_size or ic == 0: raw_i = b''; tris = []
        else:
            raw_i = data[idx_s:idx_e]; ivs = []
            for i in range(0, len(raw_i), 4):
                if i+4<=len(raw_i):
                    try: ivs.append(struct.unpack('<I', raw_i[i:i+4])[0])
                    except: pass
            tris = [tuple(ivs[i:i+3]) for i in range(0, len(ivs)-2, 3)]
        ei = {'filepath':filepath,'filename':filename,'file_size':file_size,'vertex_count':len(vbuf),
              'vertex_start':vs,'vertex_bytes':vb,'gap':gap,'index_count':ic}
        if return_data_only: return (vbuf,ubuf,tris,raw_v,raw_uv,raw_i,ei,rn,False,None)
        out = create_output_folder(filepath, name_no_ext, version)
        save_results(out, name_no_ext, raw_v, raw_uv, raw_i, vbuf, ubuf, tris, extra_info=ei, is_special=True, extra_gap=eg)
        return out

    p01 = find_first_01(data)
    if p01 is None: return None
    vip = p01+45
    if vip+4 > file_size: return None
    vc = struct.unpack('<I', data[vip:vip+4])[0]; vb = vc*16; VS = 0x9d
    if VS+vb > file_size: return None
    raw_v = data[VS:VS+vb]
    vbuf = []
    for i in range(0, vb, 16):
        c = raw_v[i:i+16]
        if len(c)>=12:
            try: x,y,z = struct.unpack('<fff', c[:12]); vbuf.append((x,y,z))
            except: pass
    gap = vb//4; ns = VS+vb; rn = data[ns:ns+gap]
    us = ns+gap; ue = us+vb
    if ue > file_size: ue = file_size
    raw_uv = data[us:ue]
    ubuf = []
    for i in range(0, len(raw_uv), 16):
        c = raw_uv[i:i+16]
        if len(c)>=8:
            try: u,v = struct.unpack('<ff', c[:8]); ubuf.append((u,v))
            except: ubuf.append((0.0,0.0))
    while len(ubuf) < len(vbuf): ubuf.append((0.0,0.0))
    if len(ubuf) > len(vbuf): ubuf = ubuf[:len(vbuf)]
    IIP = 0x75
    if IIP+4 > file_size: return None
    ic = struct.unpack('<I', data[IIP:IIP+4])[0]; ib = ic*4
    idx_s = ue; idx_e = idx_s+ib
    if idx_e > file_size: idx_e = file_size
    raw_i = data[idx_s:idx_e]; ivs = []
    for i in range(0, len(raw_i), 4):
        if i+4<=len(raw_i):
            try: ivs.append(struct.unpack('<I', raw_i[i:i+4])[0])
            except: pass
    tris = [tuple(ivs[i:i+3]) for i in range(0, len(ivs)-2, 3)]
    ei = {'filepath':filepath,'filename':filename,'file_size':file_size,'vertex_count':len(vbuf),
          'vertex_start':VS,'vertex_bytes':vb,'gap':gap,'index_count':ic,
          'vertex_info_pos':vip,'index_info_pos':IIP}
    if return_data_only: return (vbuf,ubuf,tris,raw_v,raw_uv,raw_i,ei,rn,False,None)
    out = create_output_folder(filepath, name_no_ext, version)
    save_results(out, name_no_ext, raw_v, raw_uv, raw_i, vbuf, ubuf, tris, extra_info=ei)
    return out


def process_header_1A(data, filepath, filename, version, return_data_only=False):
    name_no_ext = os.path.splitext(filename)[0]; file_size = len(data)
    VCO = 0x66; ICO = 0x6A; VS = 0x92
    if VCO+4 > file_size or ICO+4 > file_size: return None
    vc = struct.unpack('<I', data[VCO:VCO+4])[0]; vb = vc*16
    ic = struct.unpack('<I', data[ICO:ICO+4])[0]; ib_bytes = ic*4
    if VS+vb > file_size: return None
    raw_v = data[VS:VS+vb]
    vbuf = []
    for i in range(0, vb, 16):
        c = raw_v[i:i+16]
        if len(c)>=12:
            try: x,y,z = struct.unpack('<fff', c[:12]); vbuf.append((x,y,z))
            except: pass
    avc = len(vbuf); gap = vb//4; ns = VS+vb; rn = data[ns:ns+gap]
    us = ns+gap; ue = us+vb
    if ue > file_size: ue = file_size
    raw_uv = data[us:ue]
    ubuf = []
    for i in range(0, len(raw_uv), 16):
        c = raw_uv[i:i+16]
        if len(c)>=8:
            try: u,v = struct.unpack('<ff', c[:8]); ubuf.append((u,v))
            except: ubuf.append((0.0,0.0))
    while len(ubuf) < avc: ubuf.append((0.0,0.0))
    if len(ubuf) > avc: ubuf = ubuf[:avc]
    fn_l = filename.lower()
    is_sp = ('anim' in fn_l or 'anc' in fn_l) and 'ancestor' not in fn_l
    if is_sp: eg = vc*8; idx_s = ue+eg
    else: eg = 0; idx_s = ue
    idx_e = idx_s+ib_bytes
    if idx_e > file_size: idx_e = file_size
    if idx_s >= file_size or ic == 0: raw_i = b''; tris = []
    else:
        raw_i = data[idx_s:idx_e]; ivs = []
        for i in range(0, len(raw_i), 4):
            if i+4<=len(raw_i):
                try: ivs.append(struct.unpack('<I', raw_i[i:i+4])[0])
                except: pass
        tris = [tuple(ivs[i:i+3]) for i in range(0, len(ivs)-2, 3)]
    ei = {'filepath':filepath,'filename':filename,'file_size':file_size,'vertex_count':vc,
          'vertex_start':VS,'vertex_bytes':vb,'gap':gap,'index_count':ic,
          'is_special':is_sp,'extra_gap':eg}
    if return_data_only: return (vbuf,ubuf,tris,raw_v,raw_uv,raw_i,ei,rn,False,None)
    out = create_output_folder(filepath, name_no_ext, version)
    save_results(out, name_no_ext, raw_v, raw_uv, raw_i, vbuf, ubuf, tris, extra_info=ei, is_special=is_sp, extra_gap=eg)
    return out


def process_header_1C(data, filepath, filename, version, return_data_only=False):
    name_no_ext = os.path.splitext(filename)[0]; file_size = len(data)
    if 0x56 > file_size: return None
    cs = struct.unpack('<I', data[0x4E:0x52])[0]; us = struct.unpack('<I', data[0x52:0x56])[0]
    if cs<=0 or us<=0 or 0x56+cs > file_size: return None
    try: dr = do_lz4_decompress(data[0x56:0x56+cs], us)
    except: return None
    ds = len(dr); VCO = 0x34; ICO = 0x38; VS = 0x60
    if VCO+4 > ds or ICO+4 > ds: return None
    vc = struct.unpack('<I', dr[VCO:VCO+4])[0]; vb = vc*16
    ic = struct.unpack('<I', dr[ICO:ICO+4])[0]; ib = ic*4
    if VS+vb > ds: return None
    raw_v = dr[VS:VS+vb]
    vbuf = []
    for i in range(0, vb, 16):
        c = raw_v[i:i+16]
        if len(c)>=12:
            try: x,y,z = struct.unpack('<fff', c[:12]); vbuf.append((x,y,z))
            except: pass
    avc = len(vbuf); gap = vb//4; ns = VS+vb; rn = dr[ns:ns+gap]
    us2 = ns+gap; ue = us2+vb
    if ue > ds: ue = ds
    raw_uv = dr[us2:ue]
    ubuf = []
    for i in range(0, len(raw_uv), 16):
        c = raw_uv[i:i+16]
        if len(c)>=8:
            try: u,v = struct.unpack('<ff', c[:8]); ubuf.append((u,v))
            except: ubuf.append((0.0,0.0))
    while len(ubuf) < avc: ubuf.append((0.0,0.0))
    if len(ubuf) > avc: ubuf = ubuf[:avc]
    fn_l = filename.lower()
    is_sp = ('anim' in fn_l or 'anc' in fn_l) and 'ancestor' not in fn_l
    if is_sp: eg = vc*8; idx_s = ue+eg
    else: eg = 0; idx_s = ue
    idx_e = idx_s+ib
    if idx_e > ds: idx_e = ds
    if idx_s >= ds or ic == 0: raw_i = b''; tris = []
    else:
        raw_i = dr[idx_s:idx_e]; ivs = []
        for i in range(0, len(raw_i), 4):
            if i+4<=len(raw_i):
                try: ivs.append(struct.unpack('<I', raw_i[i:i+4])[0])
                except: pass
        tris = [tuple(ivs[i:i+3]) for i in range(0, len(ivs)-2, 3)]
    ei = {'filepath':filepath,'filename':filename,'file_size':file_size,
          'decompressed_size':ds,'vertex_count':vc,'vertex_start':VS,'vertex_bytes':vb,
          'gap':gap,'index_count':ic,'is_special':is_sp,'extra_gap':eg}
    if return_data_only: return (vbuf,ubuf,tris,raw_v,raw_uv,raw_i,ei,rn,True,dr)
    out = create_output_folder(filepath, name_no_ext, version)
    save_results(out, name_no_ext, raw_v, raw_uv, raw_i, vbuf, ubuf, tris,
                 extra_info=ei, is_special=is_sp, extra_gap=eg, decompressed_raw=dr)
    return out


def process_header_1E(data, filepath, filename, version, return_data_only=False):
    name_no_ext = os.path.splitext(filename)[0]; file_size = len(data)
    if 0x56 > file_size: return None
    cs = struct.unpack('<I', data[0x4E:0x52])[0]; us = struct.unpack('<I', data[0x52:0x56])[0]
    if cs<=0 or us<=0 or 0x56+cs > file_size: return None
    try: dr = do_lz4_decompress(data[0x56:0x56+cs], us)
    except: return None
    ds = len(dr)
    if ds < 0x84: return None
    svc = struct.unpack('<I', dr[0x74:0x78])[0]
    tvc = struct.unpack('<I', dr[0x78:0x7C])[0]
    pc = struct.unpack('<I', dr[0x80:0x84])[0]
    vst = 0xB3; vb = svc*16
    if vst+vb > ds: return None
    raw_v = dr[vst:vst+vb]
    vbuf = []
    for i in range(0, vb, 16):
        c = raw_v[i:i+16]
        if len(c)>=12:
            try: x,y,z = struct.unpack('<fff', c[:12]); vbuf.append((x,y,z))
            except: pass
    avc = len(vbuf)
    fn_l = filename.lower()
    is_sp = ('anim' in fn_l) or ('anc' in fn_l and 'ancestor' not in fn_l)
    if is_sp:
        gap = vb//4; ns = vst+vb; rn = dr[ns:ns+gap]; us2 = ns+gap; uvsz = vb
        eg = svc*8; idx_s = us2+uvsz+eg
    else:
        gap = svc*4-4; ns = vst+vb; rn = dr[ns:ns+gap]; us2 = ns+gap; uvsz = svc*16
        idx_s = us2+uvsz+4; eg = 0
    fc = tvc//3; idxb = fc*6; idx_e = idx_s+idxb
    if idx_e > ds: idx_e = ds
    ue = us2+uvsz
    if ue > ds: ue = ds
    raw_uv = dr[us2:ue]
    ubuf = []; p = 0
    while p+16 <= len(raw_uv):
        c = raw_uv[p:p+16]
        try: u,v = struct.unpack('<ee', c[4:8]); ubuf.append((float(u),float(v)))
        except: ubuf.append((0.0,0.0))
        p += 16
    while len(ubuf) < avc: ubuf.append((0.0,0.0))
    if len(ubuf) > avc: ubuf = ubuf[:avc]
    if idx_s >= ds or fc == 0: raw_i = b''; tris = []
    else:
        raw_i = dr[idx_s:idx_e]; ivs = []
        for i in range(0, len(raw_i), 2):
            if i+2<=len(raw_i):
                try: ivs.append(struct.unpack('<H', raw_i[i:i+2])[0])
                except: pass
        tris = [tuple(ivs[i:i+3]) for i in range(0, len(ivs)-2, 3)]
    ei = {'filepath':filepath,'filename':filename,'file_size':file_size,
          'decompressed_size':ds,'vertex_count':svc,'vertex_start':vst,'vertex_bytes':vb,
          'gap':gap,'index_count':fc*3,'is_special':is_sp,'extra_gap':eg}
    if return_data_only: return (vbuf,ubuf,tris,raw_v,raw_uv,raw_i,ei,rn,True,dr)
    out = create_output_folder(filepath, name_no_ext, version)
    save_results(out, name_no_ext, raw_v, raw_uv, raw_i, vbuf, ubuf, tris,
                 extra_info=ei, is_special=is_sp, extra_gap=eg, decompressed_raw=dr)
    return out


@dataclass
class BoneInfo:
    name: str; parent: int; matrix: list
@dataclass
class MeshData20:
    verts: list; faces: list; uv_layers: list; weights: Optional[list]


def parse_container_1F(fb):
    r = Reader(fb); hdr = r.read_fmt("<18IH"); extra = r.read_fmt("<3I")
    h = hdr[17:] + extra; csz = int(h[3]); usz = int(h[4]); bf = int(h[1])
    cds = r.tell(); comp = r.read_bytes(csz); bds = r.tell()
    bar = fb[bds:]; payload = lz4_block_decompress(comp, usz)
    bones = []
    if bf == 1:
        bi = r.read_fmt("<20I"); b = r.read_u8(); ti = r.read_u32()
        bc = int((bi+(b,ti))[17])
        for x in range(bc):
            nr = r.read_bytes(64); nm = nr.split(b"\x00",1)[0].decode("ascii",errors="ignore") or f"bone_{x}"
            mb = r.read_bytes(64); vals = struct.unpack("<16f", mb); pr = int(r.read_u32())-1
            bones.append(BoneInfo(name=nm, parent=pr, matrix=vals))
        bar = fb[bds:]
    return payload, bones, cds, csz, bf, bar


def parse_container_20(fb):
    r = Reader(fb); hdr = r.read_fmt("<18IH"); extra = r.read_fmt("<4I")
    h = hdr[17:] + extra; csz = int(h[4]); usz = int(h[5]); bf = int(h[1])
    cds = r.tell(); comp = r.read_bytes(csz); bds = r.tell()
    bar = fb[bds:]; payload = lz4_block_decompress(comp, usz)
    bones = []
    if bf == 1:
        bi = r.read_fmt("<20I"); b = r.read_u8(); ti = r.read_u32()
        bc = int((bi+(b,ti))[17])
        for x in range(bc):
            nr = r.read_bytes(64); nm = nr.split(b"\x00",1)[0].decode("ascii",errors="ignore") or f"bone_{x}"
            mb = r.read_bytes(64); vals = struct.unpack("<16f", mb); pr = int(r.read_u32())-1
            bones.append(BoneInfo(name=nm, parent=pr, matrix=vals))
        bar = fb[bds:]
    return payload, bones, cds, csz, bf, bar


def parse_standard_mesh_1F20(payload, bones):
    r = Reader(payload)
    r.seek(116); vnum = r.read_u32()
    r.seek(120); inum = r.read_u32()
    VBS = 179; r.seek(VBS)
    vbuf = r.read_bytes(vnum*16); normals = r.read_bytes(vnum*4)
    uvbuf = r.read_bytes(vnum*16)
    wbuf = r.read_bytes(vnum*8) if bones else b''
    ibuf = r.read_bytes(inum*2); ap = r.tell()
    verts = []
    for i in range(vnum):
        x,y,z = struct.unpack_from("<3f", vbuf, i*16); verts.append((x,y,z))
    idx = struct.unpack("<"+"H"*inum, ibuf)
    faces = [(idx[t*3],idx[t*3+1],idx[t*3+2]) for t in range(inum//3)]
    uv_layers = [[],[],[],[]]
    for i in range(vnum):
        uvs = struct.unpack_from("<8e", uvbuf, i*16)
        for l in range(4): uv_layers[l].append((uvs[l*2], uvs[l*2+1]))
    weights = None
    if wbuf:
        bm = list(range(-1, len(bones))); bm[0] = 0; weights = []
        for i in range(vnum):
            base = i*8; idxs = list(wbuf[base:base+4]); ws = list(wbuf[base+4:base+8])
            bids = []; bws = []
            for j in range(4):
                fw = ws[j]/255.0
                if fw<=0 or idxs[j]>=len(bm): continue
                bi = bm[idxs[j]]
                if bi<0 or bi>=len(bones): continue
                bids.append(bi); bws.append(fw)
            s = sum(bws)
            if s>0: bws = [w/s for w in bws]
            weights.append((bids, bws))
    return MeshData20(verts=verts,faces=faces,uv_layers=uv_layers,weights=weights), vbuf, normals, uvbuf, wbuf, ibuf, ap


def process_header_1F(data, filepath, filename, version, return_data_only=False):
    nn = os.path.splitext(filename)[0]; fs = len(data)
    try: payload, bones, cds, csz, bf, bar = parse_container_1F(data)
    except Exception as e: print(f"1F解析失败: {e}"); return None
    ds = len(payload); print(f"解压: {ds} 字节")
    try: md, rv, rn, ru, rw, ri, ap = parse_standard_mesh_1F20(payload, bones)
    except Exception as e: print(f"1F mesh失败: {e}"); return None
    vb = md.verts; ub = md.uv_layers[0]; tr = md.faces; vn = len(vb)
    ei = {'filepath':filepath,'filename':filename,'file_size':fs,'decompressed_size':ds,
          'vertex_count':vn,'gap':vn*4,'index_count':len(ri)//2,
          'bones':[f"{b.name} parent {b.parent}" for b in bones],'has_bones':len(bones)>0}
    if return_data_only: return (vb,ub,tr,rv,ru,ri,ei,rn,True,payload)
    out = create_output_folder(filepath, nn, version)
    save_results(out, nn, rv, ru, ri, vb, ub, tr, extra_info=ei, decompressed_raw=payload)
    return out


def process_header_20(data, filepath, filename, version, return_data_only=False):
    nn = os.path.splitext(filename)[0]; fs = len(data)
    try: payload, bones, cds, csz, bf, bar = parse_container_20(data)
    except Exception as e: print(f"20解析失败: {e}"); return None
    ds = len(payload); print(f"解压: {ds} 字节")
    try: md, rv, rn, ru, rw, ri, ap = parse_standard_mesh_1F20(payload, bones)
    except Exception as e: print(f"20 mesh失败: {e}"); return None
    vb = md.verts; ub = md.uv_layers[0]; tr = md.faces; vn = len(vb)
    ei = {'filepath':filepath,'filename':filename,'file_size':fs,'decompressed_size':ds,
          'vertex_count':vn,'gap':vn*4,'index_count':len(ri)//2,
          'bones':[f"{b.name} parent {b.parent}" for b in bones],'has_bones':len(bones)>0}
    if return_data_only: return (vb,ub,tr,rv,ru,ri,ei,rn,True,payload)
    out = create_output_folder(filepath, nn, version)
    save_results(out, nn, rv, ru, ri, vb, ub, tr, extra_info=ei, decompressed_raw=payload)
    return out


def process_single_file(filepath, is_batch=False):
    if not filepath.lower().endswith('.mesh'): return None
    try:
        with open(filepath, 'rb') as f: data = f.read()
    except: return None
    if len(data) < 4: return None
    h = data[:4]; fn = os.path.basename(filepath)
    print(f"\n处理: {fn}, {len(data)}字节, 头:{h.hex()}")
    v = HEADER_VERSION_MAP.get(h)
    if v is None: print(f"未知头: {h.hex()}"); return None
    if h in (b'\x17\x00\x00\x00',b'\x18\x00\x00\x00'):
        return process_header_17(data, filepath, fn, v, is_batch)
    elif h in (b'\x19\x00\x00\x00',b'\x1a\x00\x00\x00',b'\x1b\x00\x00\x00'):
        return process_header_1A(data, filepath, fn, v)
    elif h in (b'\x1c\x00\x00\x00',b'\x1d\x00\x00\x00'):
        return process_header_1C(data, filepath, fn, v)
    elif h == b'\x1e\x00\x00\x00':
        return process_header_1E(data, filepath, fn, v)
    elif h == b'\x1f\x00\x00\x00':
        return process_header_1F(data, filepath, fn, v)
    elif h == b'\x20\x00\x00\x00':
        return process_header_20(data, filepath, fn, v)
    return None


# ======================= OBJ解析 =======================

def parse_obj_file(obj_path):
    verts=[]; uvs=[]; fv=[]; ft=[]
    with open(obj_path,'r',encoding='utf-8',errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line[0]=='#': continue
            p = line.split()
            if not p: continue
            if p[0]=='v' and len(p)>=4: verts.append((float(p[1]),float(p[2]),float(p[3])))
            elif p[0]=='vt' and len(p)>=3: uvs.append((float(p[1]),float(p[2])))
            elif p[0]=='f':
                fvi=[]; fti=[]
                for s in p[1:]:
                    ids = s.split('/')
                    fvi.append(int(ids[0])-1)
                    fti.append(int(ids[1])-1 if len(ids)>=2 and ids[1]!='' else -1)
                for i in range(1, len(fvi)-1):
                    fv.append((fvi[0],fvi[i],fvi[i+1]))
                    ft.append((fti[0],fti[i],fti[i+1]))
    if not fv: return verts, uvs, []
    need_exp = False
    has_uv = uvs and any(t!=-1 for tri in ft for t in tri)
    if has_uv:
        for a,b in zip(fv,ft):
            for vi,ti in zip(a,b):
                if vi!=ti and ti!=-1: need_exp=True; break
            if need_exp: break
    if not need_exp:
        uo = uvs[:len(verts)] if has_uv else [(0.0,0.0)]*len(verts)
        while len(uo)<len(verts): uo.append((0.0,0.0))
        return verts, uo, fv
    nv=[]; nu=[]; nf=[]; vm={}
    for a,b in zip(fv,ft):
        tri=[]
        for vi,ti in zip(a,b):
            k=(vi,ti)
            if k not in vm:
                vm[k]=len(nv); nv.append(verts[vi])
                nu.append(uvs[ti] if 0<=ti<len(uvs) else (0.0,0.0))
            tri.append(vm[k])
        nf.append(tuple(tri))
    return nv, nu, nf


# ======================= 构建二进制工具 =======================

def build_vertex_bytes(verts):
    b = bytearray()
    for x,y,z in verts: b += struct.pack('<fff',x,y,z) + b'\x00\x00\x00\x00'
    return bytes(b)

def build_uv_float(uvs):
    b = bytearray()
    for u,v in uvs: b += struct.pack('<ff',u,v) + b'\x00'*8
    return bytes(b)

def build_uv_half_1E(uvs):
    b = bytearray()
    for u,v in uvs: b += b'\x00'*4 + struct.pack('<ee',u,v) + b'\x00'*8
    return bytes(b)

def build_uv_half_1F20(uvs):
    b = bytearray()
    for u,v in uvs: b += struct.pack('<ee',u,v) + b'\x00'*12
    return bytes(b)

def build_idx32(faces):
    b = bytearray()
    for tri in faces:
        for i in tri: b += struct.pack('<I',i)
    return bytes(b)

def build_idx16(faces):
    b = bytearray()
    for tri in faces:
        for i in tri: b += struct.pack('<H',i)
    return bytes(b)

def adj_size(orig, new_sz):
    if new_sz<=0: return b''
    ol = len(orig)
    if ol==0: return b'\x00'*new_sz
    if ol==new_sz: return orig
    if ol>new_sz: return orig[:new_sz]
    r = bytearray()
    while len(r)<new_sz: r += orig[:new_sz-len(r)]
    return bytes(r[:new_sz])


# ======================= 打包函数（最终修复版） =======================

def repack_payload_1F20(payload, old_vn, old_in, new_vn, new_in, has_bones,
                        new_vdata, new_ndata, new_uvdata, new_wdata, new_idata):
    """
    精确重建1F/20 payload。
    只修改偏移116(vnum)和120(inum)，其他header字节完全保留原样。
    然后拼接新的buffer数据 + 原始trailing。
    """
    VBS = 179

    # 完整复制原始header，只改两个count字段
    hdr = bytearray(payload[:VBS])
    struct.pack_into('<I', hdr, 116, new_vn)  # vnum
    struct.pack_into('<I', hdr, 120, new_in)  # inum
    # 偏移128及其他所有字段保持原样不动!

    # 跳过原始buffers找到trailing数据
    old_body = old_vn*16 + old_vn*4 + old_vn*16
    if has_bones: old_body += old_vn*8
    old_body += old_in*2
    trailing_start = VBS + old_body
    trailing = payload[trailing_start:] if trailing_start < len(payload) else b''

    new_payload = bytes(hdr) + new_vdata + new_ndata + new_uvdata + new_wdata + new_idata + trailing

    new_body = new_vn*16 + new_vn*4 + new_vn*16
    if has_bones: new_body += new_vn*8
    new_body += new_in*2

    print(f"  Payload重建:")
    print(f"    Header: {VBS}字节 (只改偏移116和120)")
    print(f"    原body: {old_body} -> 新body: {new_body}")
    print(f"    原vnum={old_vn} inum={old_in} -> 新vnum={new_vn} inum={new_in}")
    print(f"    Trailing: {len(trailing)}字节")
    print(f"    原payload: {len(payload)} -> 新payload: {len(new_payload)}")

    # 诊断: 检查偏移128的值是否被意外修改
    orig_128 = struct.unpack_from('<I', payload, 128)[0]
    new_128 = struct.unpack_from('<I', new_payload, 128)[0]
    print(f"    偏移128: 原始={orig_128} 新={new_128} {'✓不变' if orig_128==new_128 else '✗被改!'}")

    return new_payload


def repack_header_17(orig, verts, uvs, faces, fn):
    fs = len(orig); sa = "StripAnim" in fn
    if sa:
        vip=0x4061; iip=0x4065; vs=0x408D
        ovc = struct.unpack('<I',orig[vip:vip+4])[0]; ovb=ovc*16; og=ovb//4
        oic = struct.unpack('<I',orig[iip:iip+4])[0]
        ns=vs+ovb; ogd=orig[ns:ns+og]; ue=ns+og+ovb
        oeg=ovc*8; oegd=orig[ue:ue+oeg]; ois=ue+oeg; oie=ois+oic*4
        trail = orig[oie:] if oie<fs else b''
        nvc=len(verts); nic=len(faces)*3; nvb=nvc*16; ng=nvb//4
        hdr=bytearray(orig[:vs])
        struct.pack_into('<I',hdr,vip,nvc); struct.pack_into('<I',hdr,iip,nic)
        return bytes(hdr)+build_vertex_bytes(verts)+adj_size(ogd,ng)+build_uv_float(uvs)+adj_size(oegd,nvc*8)+build_idx32(faces)+trail
    p01=find_first_01(orig)
    if p01 is None: return None
    vip=p01+45; IIP=0x75; VS=0x9D
    ovc=struct.unpack('<I',orig[vip:vip+4])[0]; ovb=ovc*16; og=ovb//4
    oic=struct.unpack('<I',orig[IIP:IIP+4])[0]
    ns=VS+ovb; ogd=orig[ns:ns+og]; ue=ns+og+ovb; oie=ue+oic*4
    trail=orig[oie:] if oie<fs else b''
    nvc=len(verts); nic=len(faces)*3; ng=(nvc*16)//4
    hdr=bytearray(orig[:VS])
    struct.pack_into('<I',hdr,vip,nvc); struct.pack_into('<I',hdr,IIP,nic)
    return bytes(hdr)+build_vertex_bytes(verts)+adj_size(ogd,ng)+build_uv_float(uvs)+build_idx32(faces)+trail


def repack_header_1A(orig, verts, uvs, faces, fn):
    fs=len(orig); VCO=0x66; ICO=0x6A; VS=0x92
    ovc=struct.unpack('<I',orig[VCO:VCO+4])[0]; ovb=ovc*16; og=ovb//4
    oic=struct.unpack('<I',orig[ICO:ICO+4])[0]
    ns=VS+ovb; ogd=orig[ns:ns+og]; ue=ns+og+ovb
    fl=fn.lower(); sp=('anim' in fl or 'anc' in fl) and 'ancestor' not in fl
    if sp: oeg=ovc*8; oegd=orig[ue:ue+oeg]; ois=ue+oeg
    else: oegd=b''; ois=ue
    oie=ois+oic*4; trail=orig[oie:] if oie<fs else b''
    nvc=len(verts); nic=len(faces)*3; ng=(nvc*16)//4
    hdr=bytearray(orig[:VS])
    struct.pack_into('<I',hdr,VCO,nvc); struct.pack_into('<I',hdr,ICO,nic)
    egd = adj_size(oegd,nvc*8) if sp else b''
    return bytes(hdr)+build_vertex_bytes(verts)+adj_size(ogd,ng)+build_uv_float(uvs)+egd+build_idx32(faces)+trail


def repack_header_1C(orig, verts, uvs, faces, fn):
    ocs=struct.unpack('<I',orig[0x4E:0x52])[0]; ous=struct.unpack('<I',orig[0x52:0x56])[0]
    dr=lz4_block_decompress(orig[0x56:0x56+ocs], ous)
    VCO=0x34; ICO=0x38; VS=0x60
    ovc=struct.unpack('<I',dr[VCO:VCO+4])[0]; ovb=ovc*16; og=ovb//4
    oic=struct.unpack('<I',dr[ICO:ICO+4])[0]
    ns=VS+ovb; ogd=dr[ns:ns+og]; ue=ns+og+ovb
    fl=fn.lower(); sp=('anim' in fl or 'anc' in fl) and 'ancestor' not in fl
    if sp: oeg=ovc*8; oegd=dr[ue:ue+oeg]; ois=ue+oeg
    else: oegd=b''; ois=ue
    oie=ois+oic*4; trail=dr[oie:]
    nvc=len(verts); nic=len(faces)*3; ng=(nvc*16)//4
    ih=bytearray(dr[:VS])
    struct.pack_into('<I',ih,VCO,nvc); struct.pack_into('<I',ih,ICO,nic)
    egd = adj_size(oegd,nvc*8) if sp else b''
    np=bytes(ih)+build_vertex_bytes(verts)+adj_size(ogd,ng)+build_uv_float(uvs)+egd+build_idx32(faces)+trail
    nc=lz4_block_compress(np)
    oh=bytearray(orig[:0x56])
    struct.pack_into('<I',oh,0x4E,len(nc)); struct.pack_into('<I',oh,0x52,len(np))
    return bytes(oh)+nc+orig[0x56+ocs:]


def repack_header_1E(orig, verts, uvs, faces, fn):
    ocs=struct.unpack('<I',orig[0x4E:0x52])[0]; ous=struct.unpack('<I',orig[0x52:0x56])[0]
    dr=lz4_block_decompress(orig[0x56:0x56+ocs], ous); ds=len(dr)
    osvc=struct.unpack('<I',dr[0x74:0x78])[0]; otvc=struct.unpack('<I',dr[0x78:0x7C])[0]
    vst=0xB3; ovb=osvc*16
    fl=fn.lower(); sp=('anim' in fl) or ('anc' in fl and 'ancestor' not in fl)
    if sp:
        og=ovb//4; ogd=dr[vst+ovb:vst+ovb+og]; ue=vst+ovb+og+ovb
        oeg=osvc*8; oegd=dr[ue:ue+oeg]; ois=ue+oeg
    else:
        og=osvc*4-4; ogd=dr[vst+ovb:vst+ovb+og]; ue=vst+ovb+og+osvc*16
        ois=ue+4; oegd=b''
    ofc=otvc//3; oie=ois+ofc*6; trail=dr[oie:]
    nvc=len(verts); nic=len(faces)*3
    ih=bytearray(dr[:vst])
    struct.pack_into('<I',ih,0x74,nvc); struct.pack_into('<I',ih,0x78,nic)
    # 偏移0x80保持原值不动!
    if sp:
        ng=(nvc*16)//4; body=build_vertex_bytes(verts)+adj_size(ogd,ng)+build_uv_half_1E(uvs)+adj_size(oegd,nvc*8)+build_idx16(faces)+trail
    else:
        ng=max(0,nvc*4-4); body=build_vertex_bytes(verts)+adj_size(ogd,ng)+build_uv_half_1E(uvs)+b'\x00'*4+build_idx16(faces)+trail
    np=bytes(ih)+body; nc=lz4_block_compress(np)
    oh=bytearray(orig[:0x56])
    struct.pack_into('<I',oh,0x4E,len(nc)); struct.pack_into('<I',oh,0x52,len(np))
    return bytes(oh)+nc+orig[0x56+ocs:]


def repack_header_1F(orig, verts, uvs, faces, fn):
    r=Reader(orig); hdr=r.read_fmt("<18IH"); extra=r.read_fmt("<3I")
    h=hdr[17:]+extra; ocsz=int(h[3]); ousz=int(h[4]); bf=int(h[1])
    cds=r.tell(); oc=r.read_bytes(ocsz); bds=r.tell(); bar=orig[bds:]
    payload=lz4_block_decompress(oc, ousz); hb=bf==1

    pr=Reader(payload); pr.seek(116); ovn=pr.read_u32(); pr.seek(120); oin=pr.read_u32()
    VBS=179; pr.seek(VBS)
    pr.read_bytes(ovn*16); pr.read_bytes(ovn*4); pr.read_bytes(ovn*16)
    if hb: pr.read_bytes(ovn*8)
    pr.read_bytes(oin*2)

    nvn=len(verts); nin=len(faces)*3
    nv=build_vertex_bytes(verts); nn=adj_size(payload[VBS+ovn*16:VBS+ovn*16+ovn*4], nvn*4)
    nu=build_uv_half_1F20(uvs)
    nw=adj_size(payload[VBS+ovn*16+ovn*4+ovn*16:VBS+ovn*16+ovn*4+ovn*16+ovn*8], nvn*8) if hb else b''
    ni=build_idx16(faces)

    np=repack_payload_1F20(payload, ovn, oin, nvn, nin, hb, nv, nn, nu, nw, ni)
    nc=lz4_block_compress(np)
    oh=bytearray(orig[:cds])
    struct.pack_into('<I', oh, cds-8, len(nc))
    struct.pack_into('<I', oh, cds-4, len(np))
    print(f"  外部header: comp@0x{cds-8:X}={len(nc)}, uncomp@0x{cds-4:X}={len(np)}")
    return bytes(oh)+nc+bar


def repack_header_20(orig, verts, uvs, faces, fn):
    r=Reader(orig); hdr=r.read_fmt("<18IH"); extra=r.read_fmt("<4I")
    h=hdr[17:]+extra; ocsz=int(h[4]); ousz=int(h[5]); bf=int(h[1])
    cds=r.tell(); oc=r.read_bytes(ocsz); bds=r.tell(); bar=orig[bds:]
    payload=lz4_block_decompress(oc, ousz); hb=bf==1

    pr=Reader(payload); pr.seek(116); ovn=pr.read_u32(); pr.seek(120); oin=pr.read_u32()
    VBS=179; pr.seek(VBS)
    pr.read_bytes(ovn*16); pr.read_bytes(ovn*4); pr.read_bytes(ovn*16)
    if hb: pr.read_bytes(ovn*8)
    pr.read_bytes(oin*2)

    nvn=len(verts); nin=len(faces)*3
    nv=build_vertex_bytes(verts); nn=adj_size(payload[VBS+ovn*16:VBS+ovn*16+ovn*4], nvn*4)
    nu=build_uv_half_1F20(uvs)
    nw=adj_size(payload[VBS+ovn*16+ovn*4+ovn*16:VBS+ovn*16+ovn*4+ovn*16+ovn*8], nvn*8) if hb else b''
    ni=build_idx16(faces)

    np=repack_payload_1F20(payload, ovn, oin, nvn, nin, hb, nv, nn, nu, nw, ni)
    nc=lz4_block_compress(np)
    oh=bytearray(orig[:cds])
    struct.pack_into('<I', oh, cds-8, len(nc))
    struct.pack_into('<I', oh, cds-4, len(np))
    print(f"  外部header: comp@0x{cds-8:X}={len(nc)}, uncomp@0x{cds-4:X}={len(np)}")
    return bytes(oh)+nc+bar


# ======================= OBJ转MESH =======================

def obj_to_mesh_process():
    print("\n=== OBJ 转 Mesh ===")
    op = input("OBJ文件路径: ").strip().strip('"').strip("'")
    if not os.path.isfile(op): print(f"不存在: {op}"); return
    mp = input("原始mesh模板路径: ").strip().strip('"').strip("'")
    if not os.path.isfile(mp): print(f"不存在: {mp}"); return

    verts, uvs, faces = parse_obj_file(op)
    if not verts or not faces: print("OBJ数据无效"); return
    while len(uvs)<len(verts): uvs.append((0.0,0.0))
    if len(uvs)>len(verts): uvs=uvs[:len(verts)]
    print(f"OBJ: 顶点={len(verts)} UV={len(uvs)} 三角={len(faces)}")

    with open(mp,'rb') as f: orig=f.read()
    if len(orig)<4: print("太小"); return
    h=orig[:4]; fn=os.path.basename(mp); v=HEADER_VERSION_MAP.get(h)
    if v is None: print(f"未知头: {h.hex()}"); return
    print(f"模板: 版本{v} ({h.hex()}) {fn}")

    result=None
    try:
        if h in (b'\x17\x00\x00\x00',b'\x18\x00\x00\x00'): result=repack_header_17(orig,verts,uvs,faces,fn)
        elif h in (b'\x19\x00\x00\x00',b'\x1a\x00\x00\x00',b'\x1b\x00\x00\x00'): result=repack_header_1A(orig,verts,uvs,faces,fn)
        elif h in (b'\x1c\x00\x00\x00',b'\x1d\x00\x00\x00'): result=repack_header_1C(orig,verts,uvs,faces,fn)
        elif h==b'\x1e\x00\x00\x00': result=repack_header_1E(orig,verts,uvs,faces,fn)
        elif h==b'\x1f\x00\x00\x00': result=repack_header_1F(orig,verts,uvs,faces,fn)
        elif h==b'\x20\x00\x00\x00': result=repack_header_20(orig,verts,uvs,faces,fn)
        else: print(f"不支持: {h.hex()}"); return
    except Exception as e:
        print(f"打包失败: {e}"); import traceback; traceback.print_exc(); return
    if result is None: print("失败"); return

    bd=os.path.dirname(os.path.abspath(mp)); nn=os.path.splitext(fn)[0]
    op2=os.path.join(bd, f"{nn}_repacked.mesh"); i=1
    while os.path.exists(op2): op2=os.path.join(bd, f"{nn}_repacked_{i}.mesh"); i+=1
    with open(op2,'wb') as f: f.write(result)
    print(f"\n成功! {op2}")
    print(f"大小: {len(result)} (原{len(orig)}), 顶点={len(verts)} 三角={len(faces)}")

    # 验证
    print("\n--- 验证 ---")
    try:
        vh=result[:4]; vv=HEADER_VERSION_MAP.get(vh); vr=None
        if vh in (b'\x17\x00\x00\x00',b'\x18\x00\x00\x00'):
            vr=process_header_17(result,op2,fn,vv,True,True)
        elif vh in (b'\x19\x00\x00\x00',b'\x1a\x00\x00\x00',b'\x1b\x00\x00\x00'):
            vr=process_header_1A(result,op2,fn,vv,True)
        elif vh in (b'\x1c\x00\x00\x00',b'\x1d\x00\x00\x00'):
            vr=process_header_1C(result,op2,fn,vv,True)
        elif vh==b'\x1e\x00\x00\x00': vr=process_header_1E(result,op2,fn,vv,True)
        elif vh==b'\x1f\x00\x00\x00': vr=process_header_1F(result,op2,fn,vv,True)
        elif vh==b'\x20\x00\x00\x00': vr=process_header_20(result,op2,fn,vv,True)
        if vr:
            print(f"解析: v={len(vr[0])} uv={len(vr[1])} t={len(vr[2])}")
            if len(vr[0])==len(verts) and len(vr[2])==len(faces): print("一致 ✓")
            else: print(f"不匹配!")

            # 额外验证: 对比payload header
            if vr[8] and vr[9]:  # is_compressed, payload
                new_pl = vr[9]
                orig_pl_bytes = None
                if vh == b'\x1f\x00\x00\x00':
                    try:
                        _,_,_,_,_,_ = parse_container_1F(orig)
                        orig_pl_bytes = parse_container_1F(orig)[0]
                    except: pass
                elif vh == b'\x20\x00\x00\x00':
                    try: orig_pl_bytes = parse_container_20(orig)[0]
                    except: pass
                if orig_pl_bytes:
                    print("\n  Payload header对比 (偏移100-178):")
                    for off in range(100, 179, 4):
                        if off+4 <= len(orig_pl_bytes) and off+4 <= len(new_pl):
                            ov = struct.unpack_from('<I', orig_pl_bytes, off)[0]
                            nv2 = struct.unpack_from('<I', new_pl, off)[0]
                            tag = ""
                            if off==116: tag="vnum"
                            elif off==120: tag="inum"
                            elif off==128: tag="field_128"
                            changed = " <-变更" if ov!=nv2 else ""
                            if ov!=0 or nv2!=0:
                                print(f"    0x{off:02X}: {ov:8d} -> {nv2:8d}  {tag}{changed}")
        else: print("验证失败")
    except Exception as e: print(f"验证出错: {e}")


# ======================= 其他功能 =======================

def auto_process():
    if len(sys.argv)>1: fn=sys.argv[1]
    else:
        try: fn=input("mesh文件路径: ").strip().strip('"').strip("'")
        except: return
    if not os.path.exists(fn): print(f"不存在: {fn}"); return
    process_single_file(fn)

def process_directory():
    dp=input("目录路径: ").strip().strip('"').strip("'")
    if not os.path.isdir(dp): print("无效"); return
    mf=[]
    for r,d,fs in os.walk(dp):
        for f in fs:
            if f.lower().endswith('.mesh'): mf.append(os.path.join(r,f))
    if not mf: print("无mesh文件"); return
    print(f"找到{len(mf)}个文件")
    for i,fp in enumerate(mf,1):
        print(f"\n[{i}/{len(mf)}] {fp}")
        process_single_file(fp, True)

def manual_process():
    print("\n=== 手动模式 ===")
    vh=""
    print("顶点hex(空行结束):")
    while True:
        l=input().strip()
        if l=="": break
        vh+=l
    vd=hex_string_to_bytes(vh)
    if not vd: return
    vb=[]; p=0
    while p+16<=len(vd):
        try: x,y,z=struct.unpack('<fff',vd[p:p+12]); vb.append((x,y,z)); p+=16
        except: p+=1
    print(f"顶点: {len(vb)}")
    uh=""
    print("UV hex(空行结束):")
    while True:
        l=input().strip()
        if l=="": break
        uh+=l
    ub=[]; ud=b''
    if uh:
        ud=hex_string_to_bytes(uh); p=0
        while p+16<=len(ud):
            try: u,v=struct.unpack('<ff',ud[p:p+8]); ub.append((u,v)); p+=16
            except: p+=1
    ih=""
    print("索引hex(空行结束):")
    while True:
        l=input().strip()
        if l=="": break
        ih+=l
    ib=[]; id2=b''
    if ih:
        id2=hex_string_to_bytes(ih); p=0; t=[]
        while p+4<=len(id2):
            try: v2=struct.unpack('<I',id2[p:p+4])[0]; t.append(v2); p+=4
            except: p+=1
            if len(t)>=3: ib.append(tuple(t[:3])); t=[]
    while len(ub)<len(vb): ub.append((0.0,0.0))
    bd=os.getcwd(); fi=1
    while True:
        of=os.path.join(bd,f"manual_output_{fi}")
        if not os.path.exists(of): os.makedirs(of); break
        fi+=1
    op=os.path.join(of,"manual_mesh.obj")
    with open(op,'w') as f:
        f.write("# Manual\n")
        for v in vb: f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for u in ub: f.write(f"vt {u[0]:.6f} {u[1]:.6f}\n")
        for tri in ib: f.write(f"f {tri[0]+1}/{tri[0]+1} {tri[1]+1}/{tri[1]+1} {tri[2]+1}/{tri[2]+1}\n")
    print(f"OBJ: {op}")


def main():
    print("=== Mesh工具 (17/18/19/1A/1B/1C/1D/1E/1F/20) ===")
    print("1. 批量处理目录")
    print("2. 处理单个文件")
    print("3. 手动输入")
    print("4. OBJ转Mesh")
    print("=" * 50)
    while True:
        try:
            c=input("选择(1/2/3/4): ").strip()
            if c=="1": process_directory(); break
            elif c=="2": auto_process(); break
            elif c=="3": manual_process(); break
            elif c=="4": obj_to_mesh_process(); break
        except KeyboardInterrupt: print("\n终止"); break
        except Exception as e: print(f"错误: {e}"); import traceback; traceback.print_exc(); break
    input("\n按Enter退出...")

if __name__ == "__main__":
    main()