# OpenCode + OpenSpec 集成方案

## 一、方案概述

将 OpenSpec 集成到项目的 OpenCode 配置中，实现规范驱动的 AI 辅助开发。

### 集成目标

- OpenSpec 命令集成（斜杠命令）
- 项目级 AGENTS.md 指令
- 配置中文输出

### 适用场景

现有代码重构治理（局部优化）

---

## 二、集成步骤

### 步骤 1：检查环境

```bash
node --version   # 需 >= 20.19.0
npm --version
```

### 步骤 2：安装 OpenSpec

```bash
npm install -g @fission-ai/openspec@latest
openspec --version
```

### 步骤 3：初始化项目

```bash
cd your-project-directory
openspec init
```

### 步骤 4：启用 Expanded Profile

```bash
openspec config profile
# 选择：Expanded Profile

openspec update
```

---

## 三、配置文件

### 3.1 opencode.json

在项目根目录创建 `opencode.json`：

```json
{
  "$schema": "https://opencode.ai/config.json",
  "instructions": [
    "AGENTS.md",
    "doc/"
  ],
  "permission": {
    "skill": {
      "*": "allow"
    }
  },
  "agent": {
    "plan": {
      "permission": {
        "skill": {
          "*": "allow"
        }
      }
    }
  }
}
```

### 3.2 AGENTS.md

在项目根目录创建 `AGENTS.md`：

```markdown
# 项目指南

## 语言要求

- 所有思考过程和输出请用中文
- 使用简体中文回复
- 保持回复简洁，除非用户要求详细说明

## 技术栈

请在此处填写项目技术栈，例如：
- Spring Boot / Next.js / Go 等

## 项目结构

请在此处描述项目结构

## 核心业务模块

请在此处列出核心业务模块

## 重构优化方向

请在此处列出重构优化方向

## 开发规范

- 使用 Given/When/Then 格式描述测试场景
- 变更必须包含回滚方案
- 标注影响的模块和数据库表
- 输出语言：中文

## OpenSpec 工作流

使用 Expanded Profile 完整命令：

| 命令 | 说明 |
|------|------|
| /opsx:new | 创建变更 |
| /opsx:ff | 快速生成所有文档 |
| /opsx:apply | 执行实现 |
| /opsx:verify | 验证一致性 |
| /opsx:archive | 归档 |
```

### 3.3 openspec/config.yaml

在 `openspec/` 目录下创建或更新 `config.yaml`：

```yaml
schema: spec-driven

context: |
  项目：{{项目名称}}
  输出语言：中文
  技术栈：{{技术栈}}
  业务模块：{{核心业务模块}}
  重构优化方向：{{重构优化方向}}

rules:
  proposal:
    - 必须使用中文输出
    - 必须包含回滚方案
    - 标注影响的模块范围
    - 说明数据库变更影响
  specs:
    - 使用中文描述测试场景
    - 标注涉及的表和API
```

---

## 四、重构工作流

### 推荐：完整路径

```
/opsx:new [变更名称]  →  /opsx:ff  →  /opsx:apply  →  /opsx:verify  →  /opsx:archive
```

### 使用示例

```
/opsx:new 描述你的变更任务
/opsx:ff
/opsx:apply
/opsx:verify
/opsx:archive
```

---

## 五、验证检查清单

```bash
# 检查 OpenSpec 安装
openspec --version

# 初始化后检查目录
ls -la .opencode/commands/

# 检查 OpenSpec 状态
openspec status

# 验证变更目录
ls -la openspec/changes/
```

---

## 六、目录结构

集成后的项目结构：

```
your-project/
├── opencode.json               # OpenCode 项目配置
├── AGENTS.md                   # 项目指令文件（包含中文要求）
├── .opencode/                  # OpenCode 目录
│   └── commands/               # OpenSpec 斜杠命令（自动生成）
├── openspec/                   # OpenSpec 目录
│   ├── config.yaml             # OpenSpec 配置（包含中文规则）
│   └── changes/                # 变更规范目录
└── doc/                        # 项目文档
```

---

## 七、相关文档

- [OpenSpec 官方文档](https://github.com/fission-ai/openspec)
