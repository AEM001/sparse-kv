# SpecExtend 本地部署日志

## 环境
- 硬件：2x RTX 3090 (24GB)
- OS: Linux, Python 3.12
- CUDA: 12.8 (Driver 580.105), PyTorch 2.4.0+cu121

---

## 01:11 开始
查看项目结构，确认需要下载的模型：
- target: `lmsys/vicuna-7b-v1.5-16k`
- draft: `double7/vicuna-68m`

## 01:12 模型下载
- 创建 `download_models_mirror.sh`，使用 `hf-mirror.com` 免代理下载
- 缓存目录定向到数据盘：`HF_HOME=/root/autodl-tmp/huggingface`
- 创建 `monitor_download.py` 监测下载进度和卡住检测

## 01:28 模型下载完成
- `lmsys/vicuna-7b-v1.5-16k`: 13.5 GB
- `double7/vicuna-68m`: 260 MB

## 01:29 虚拟环境与依赖
- 在 `specextend/` 下创建 `.venv`
- 系统盘空间不足（30GB），确认 .venv 在代码目录下，模型在数据盘

## 01:32 网络加速安装
- 使用 `source /etc/network_turbo` 加速 pip 下载
- 安装 torch==2.4.0+cu121（cached，--no-deps 方式先装核心）

## 01:45 预编译 flash-attn
- 查找预编译 wheel，找到 `mjun0812/flash-attention-prebuild-wheels`
- 下载 `flash_attn-2.6.3+cu121torch2.4-cp312-cp312-linux_x86_64.whl`（178MB）

## 01:50 依赖安装完成
- flash-attn 2.6.3 安装成功（免编译）
- transformers 4.41.0、accelerate 0.21.0 等全部就绪
- 验证：`torch.cuda.is_available() == True`

## 01:55 推理遇到问题
- 运行 `run_classic.py` 报错：`ConnectionError` 连 huggingface.co
- 原因：transformers 默认会联网检查 safetensors/index

## 01:57 离线加载修复
- `classic/model_classic.py:28`：`AutoTokenizer.from_pretrained(..., local_files_only=True)`
- `classic/model_classic.py:46-51`：`KVLlamaForCausalLM.from_pretrained(..., local_files_only=True, **kwargs)`

## 01:59 本地缓存路径解析
- `run_classic.py` 添加 `resolve_local_path()`：自动将 model_id 映射到本地 snapshot 目录
- 移除 `device_map="auto"`，改为 `.eval().cuda()`（避免跨 GPU 问题）

## 02:00 推理跑通
- 2K 输入 + 256 tokens 生成成功
- Baseline 和 SpecExtend 均正常运行

## 02:01 README 更新
- 补充完整本地部署指南：venv、torch 安装、flash-attn 预编译 wheel、模型下载、离线推理
- 添加性能对比表格和项目文件说明

## 02:03 4K 对比测试
| 方法 | 输入长度 | 生成长度 | 耗时 |
|------|---------|---------|------|
| Baseline | 4K | 128 | 17.09 s |
| SpecExtend | 4K | 128 | 17.81 s |

- 8K 测试 OOM（24GB 显存不足）

## 02:05 短序列慢的原因分析
SpecExtend 在 2K-4K 上无加速甚至略慢，核心原因：

1. **draft 接受率低**：短序列上 68M draft 和 7B target 对齐差，生成的候选 token 大部分被 reject，等于白白多跑一个模型前向。
2. **retrieval overhead**：每 8 步触发一次 cross-model retrieval（取 attention scores → chunk 聚合 → top-k 选取 → KV cache 重组），这些 GPU 操作在短序列上占比大。
3. **flash-attn 短序列优势小**：2K-4K 的 attention 计算量本身不大，kernel launch 固定开销占比高。
4. **tree attention mask 额外计算**：speculative decoding 的树形 mask 比普通 causal mask 复杂。

论文声称的 **2.84x 加速是在 16K 长度 + A100 80GB** 上，长序列下 draft 接受率更高，flash-attention 和 retrieval 的收益才能盖过 overhead。

---

## 结论
- 项目已成功本地部署并跑通
- 2K-4K 短序列上 SpecExtend ≈ Baseline（无显著加速）
- 8K+ 需要更大显存（建议 40GB+）才能测试
