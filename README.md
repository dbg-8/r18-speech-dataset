# R18 Speech Dataset
NSFW  违规中文声音数据集

**注意**：谨慎使用 - 办公室及公共场合请带上耳机

违规声音数据集是一个由短人声组成的音频数据集，这些人声从网上公开有声读物及某些众所周知的网站付费下载的有声读物的录音中抽取。可用于语音合规审查的分类任务，包含下面一些内容：

- 娇喘呓语
- 常规话语
- 色情内容
- 粗口改编歌曲

为了收集我们的所有数据，我们通过付费会员的方式获得在众所周知网站的音频下载资格，基于这些网站的内容简介圈定待定数据，最后通过人工验证的方式逐段确认筛选标注最终形成完成数据集。

本数据集有几个特点：

- 大陆普通话，100+ 讲话人，1000 Hours+ 话语
- 来源包含 有声读物、主播短视频、各短视频网站， 清晰无噪音
- 多数片段涉及人工验证
- 数据经过脱敏， 不包含任何与确认自然人有关的信息
- 讲话人多为众多播主

数据说明


- 10s, 16KHz, Mono wav 格式。`{label}/{ID}-{sn}.wav`
- ansfw.csv 元数据
- ansfw.scp.txt 对应自动语音识别文本
- samples/{label}/**.wav 样例，每类别100条

下载数据在[这里](https://aistudio.baidu.com/aistudio/datasetdetail/107582)

> 本数据集遵守 CC-BY 4.0 协议。作者 `火工道人` 放弃作品的版权和其他相关权利。

