# 檔名: build_engine_fixed.py
import tensorrt as trt
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Build a TensorRT engine from an ONNX file.")
    parser.add_argument("--onnx", required=True, help="Path to the ONNX model file.")
    parser.add_argument("--engine", required=True, help="Path to save the TensorRT engine file.")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mode.")
    parser.add_argument("--workspace", type=int, default=1024, help="Workspace size in MiB.")
    parser.add_argument("--batch_size", type=int, default=1, help="Fixed batch size.")
    parser.add_argument("--height", type=int, default=200, help="Input height.")
    parser.add_argument("--width", type=int, default=200, help="Input width.")
    opts = parser.parse_args()

    if not os.path.exists(opts.onnx):
        print(f"❌ ONNX 檔案不存在: {opts.onnx}")
        return

    print("🚀 開始構建 TensorRT 引擎...")
    print(f"📁 ONNX: {opts.onnx}")
    print(f"📁 Engine: {opts.engine}")
    print(f"🔧 精度模式: {'FP16' if opts.fp16 else 'FP32'}")
    print(f"🔧 輸入形狀: ({opts.batch_size}, 3, {opts.height}, {opts.width})")

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)  # 更詳細的日誌
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser_trt = trt.OnnxParser(network, TRT_LOGGER)

    # 解析 ONNX 模型
    print(f"📖 載入 ONNX 檔案...")
    with open(opts.onnx, 'rb') as model:
        if not parser_trt.parse(model.read()):
            print("❌ ONNX 解析失敗:")
            for error in range(parser_trt.num_errors):
                print(f"   {parser_trt.get_error(error)}")
            return
    print(f"✅ ONNX 檔案解析成功")
    
    # 建立配置
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, opts.workspace * (1024 * 1024))
    
    # 🔥 關鍵修復：更嚴格的 optimization profile 設定
    profile = builder.create_optimization_profile()
    
    # 獲取輸入名稱
    input_name = network.get_input(0).name
    print(f"🔍 輸入層名稱: {input_name}")
    
    # 設定固定形狀（避免動態形狀問題）
    input_shape = (opts.batch_size, 3, opts.height, opts.width)
    profile.set_shape(input_name, 
                      min=input_shape,    # 最小 = 最佳 = 最大，都一樣
                      opt=input_shape,    
                      max=input_shape)    
        
    config.add_optimization_profile(profile)
    print(f"✅ Optimization profile 設定完成")
    
    # 設定精度
    if opts.fp16:
        if builder.platform_has_fast_fp16:
            print("🚀 啟用 FP16 模式")
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("⚠️  硬體不支援 FP16，回退到 FP32")
    else:
        print("🚀 使用 FP32 模式")
    
    # 建立引擎
    print("🔨 開始構建 TensorRT 引擎... (這可能需要幾分鐘)")
    
    try:
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            print("❌ 引擎構建失敗")
            return
            
        # 保存引擎
        with open(opts.engine, 'wb') as f:
            f.write(serialized_engine)
            
        print(f"✅ 引擎構建成功!")
        print(f"📁 已保存至: {opts.engine}")
        print(f"📊 檔案大小: {os.path.getsize(opts.engine) / (1024*1024):.2f} MB")
        
        # 驗證引擎
        print("\n🔍 驗證引擎...")
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        
        if engine is None:
            print("❌ 引擎驗證失敗")
            return
            
        print(f"✅ 引擎驗證成功")
        print(f"🔍 輸入數量: {engine.num_bindings // 2}")
        print(f"🔍 輸出數量: {engine.num_bindings // 2}")
        
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            is_input = engine.binding_is_input(i)
            shape = engine.get_binding_shape(i)
            dtype = engine.get_binding_dtype(i)
            print(f"   {'輸入' if is_input else '輸出'} {i}: {name}, 形狀={shape}, 類型={dtype}")
        
    except Exception as e:
        print(f"❌ 構建過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()