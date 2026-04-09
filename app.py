import os
import tempfile
import mesh_converter  # 导入您的完整脚本
from flask import Flask, request, send_file
from flask_cors import CORS
import io

app = Flask(__name__)
CORS(app)

@app.route('/convert', methods=['POST'])
def convert():
    try:
        # 获取上传的文件
        obj_file = request.files['obj_file']
        mesh_template = request.files['mesh_template']
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.obj') as tmp_obj:
            obj_file.save(tmp_obj.name)
            obj_path = tmp_obj.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mesh') as tmp_tpl:
            mesh_template.save(tmp_tpl.name)
            tpl_path = tmp_tpl.name
        
        # 解析OBJ文件（使用您脚本中的 parse_obj_file）
        verts, uvs, faces = mesh_converter.parse_obj_file(obj_path)
        
        # 读取模板文件
        with open(tpl_path, 'rb') as f:
            tpl_data = f.read()
        
        # 获取模板头版本
        header = tpl_data[:4]
        fn = os.path.basename(tpl_path)
        
        # 根据版本调用对应的打包函数
        if header in (b'\x17\x00\x00\x00', b'\x18\x00\x00\x00'):
            result = mesh_converter.repack_header_17(tpl_data, verts, uvs, faces, fn)
        elif header in (b'\x19\x00\x00\x00', b'\x1a\x00\x00\x00', b'\x1b\x00\x00\x00'):
            result = mesh_converter.repack_header_1A(tpl_data, verts, uvs, faces, fn)
        elif header in (b'\x1c\x00\x00\x00', b'\x1d\x00\x00\x00'):
            result = mesh_converter.repack_header_1C(tpl_data, verts, uvs, faces, fn)
        elif header == b'\x1e\x00\x00\x00':
            result = mesh_converter.repack_header_1E(tpl_data, verts, uvs, faces, fn)
        elif header == b'\x1f\x00\x00\x00':
            result = mesh_converter.repack_header_1F(tpl_data, verts, uvs, faces, fn)
        elif header == b'\x20\x00\x00\x00':
            result = mesh_converter.repack_header_20(tpl_data, verts, uvs, faces, fn)
        else:
            return f'不支持的文件头: {header.hex()}', 400
        
        # 清理临时文件
        os.unlink(obj_path)
        os.unlink(tpl_path)
        
        # 返回结果
        return send_file(
            io.BytesIO(result),
            as_attachment=True,
            download_name='repacked.mesh',
            mimetype='application/octet-stream'
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return str(e), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))