import os
import sys

try:
    from gguf import GGUFReader, GGUFWriter, GGUFValueType
except ImportError:
    print("❌ 'gguf' 라이브러리가 없습니다. 'pip install gguf'를 실행해주세요.")
    sys.exit(1)

input_file = "EXAONE-3.5-2.4B-Instruct.BF16.gguf"
output_file = "EXAONE-3.5-2.4B-Instruct.FIXED.gguf"

if not os.path.exists(input_file):
    print(f"❌ {input_file} 파일이 없습니다.")
    sys.exit(1)

print(f"🛠️ {input_file} 메타데이터 수정 시작...")

reader = GGUFReader(input_file)
writer = GGUFWriter(output_file, arch="exaone")

for kv in reader.fields.values():
    key = kv.name
    # llama.cpp가 요구하는 이름으로 변경
    effective_key = "exaone.attention.layer_norm_rms_epsilon" if key == "exaone.attention.layer_norm_epsilon" else key
    
    if key == "exaone.attention.layer_norm_epsilon":
        print(f"✅ 변경: {key} -> {effective_key}")

    val_type = kv.types[0]
    data = kv.data

    # 타입별로 적절한 add_* 메서드 호출
    if val_type == GGUFValueType.UINT32:
        writer.add_uint32(effective_key, int(data[0]))
    elif val_type == GGUFValueType.INT32:
        writer.add_int32(effective_key, int(data[0]))
    elif val_type == GGUFValueType.FLOAT32:
        writer.add_float32(effective_key, float(data[0]))
    elif val_type == GGUFValueType.BOOL:
        writer.add_bool(effective_key, bool(data[0]))
    elif val_type == GGUFValueType.STRING:
        s = str(data[0]) if not isinstance(data[0], (bytes, bytearray)) else bytes(data[0]).decode('utf-8', errors='ignore')
        writer.add_string(effective_key, s)
    elif val_type == GGUFValueType.ARRAY:
        writer.add_array(effective_key, data)
    else:
        # 기타 타입은 일단 필드 추가 시도 (버전에 따라 다를 수 있음)
        try:
            writer.add_field(effective_key, data, kv.types)
        except AttributeError:
            print(f"⚠️ 건너뜀 (지원되지 않는 타입): {key}")

# 텐서 정보 복사
print("🔄 텐서 정보 복사 중...")
for tensor in reader.tensors:
    writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype, tensor.tensor_type)

print("💾 파일 저장 중... (약 1~2분)")
writer.write_config_to_file()
writer.write_tensors_to_file(input_file)

print(f"🎉 수정 완료! 이제 {output_file} 파일로 양자화하세요.")