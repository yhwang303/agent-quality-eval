# 安装 agent-quality-eval / agent-cot hook 后需要重启 IDE 吗

需要，建议重启对应 IDE。hook 配置、runtime 路径和脚本资产通常在 IDE 进程启动时加载。推荐关闭 IDE，安装并重新 init hook，重启 IDE，再开启新会话验证 trace/critic 是否生成。
