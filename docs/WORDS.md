# 词表与候选词

`data/source/wordData.js` 是人造语词典（`dh` = 罗马字拼写，`cn` = 中文释义，`type` = 词性），
本工程把它接入手写输入法，做两件事：

1. **候选词**：边写边给出以当前前缀开头的词，点一下补全
2. **反向约束单字**：当前前缀是 `to` 时，只有能构成合法词的字母（`n`→`ton`）才被优先，
   `m` 会因为在词表里不存在 `tom` 而被压下去

## 转换步骤

`wordData.js` 是带模板字符串的 JS，不要用正则硬啃。用 Node 求值后转成 JSON：

```bat
node -e "const fs=require('fs');eval(fs.readFileSync('data/source/wordData.js','utf8'));fs.writeFileSync('data/words.json',JSON.stringify(globalThis.wordData))"
```

当前 `data/words.json`：**2818 条**，`dh` 长度 1–15。

## 字符覆盖

词表用到但模型 100 类**不支持**的字符：`ô`（13 次）、`‘`（1 次）、`’`（2 次），
共影响 16/2818 条（0.57%）。这些词仍会出现在候选里（可以点选补全），但不参与单字反向约束。

若要完全覆盖，需要给字符集加 `ô Ô` 并重新采集这两类的样本。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/suggest?prefix=to&limit=10` | 返回以 `to` 开头的词（完全相等优先，其次短词优先） |
| POST | `/api/recognize` | 请求体可带 `prefix`，服务端据此做单字反向约束 |

## 前端行为

* 结束连续字母后自动查询候选词（120 ms 防抖）
* 点击候选词 → 补全剩余字母（前缀 `to` + 点 `tonh` → 追加 `nh`）
* `空格` / `退格` / `清空` 都会刷新候选词
* 采集模式下不显示候选词（避免干扰标注）

## 调试入口

* `http://127.0.0.1:8765/?text=to` 预填文本并展示候选词
* `http://127.0.0.1:8765/?mode=collect` 直接进采集模式
