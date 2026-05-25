# OpenCode + OpenSpec + OpenPencil 集成方案

## 一、方案概述

将 OpenPencil（AI 原生矢量设计工具）集成到 OpenSpec 的生命周期中，遵循 **需求(specs) → 架构(design) → UI(design-ui) → 前后端(tasks)** 的最佳实践。

### 整体架构

```
OpenSpec + OpenPencil 工作流（需求 → 架构 → UI → 前后端）：

用户请求 → brainstorm → proposal → specs
                                    ↓
                               design（含前端架构 + 设计令牌）
                                    ↓
                               design-ui（.op 设计文件）
                                    ↓
                               tasks（引用 .op + specs）
                                    ↓
                               plan → apply → verify → retrospective
```

### 三者定位

| 工具 | 定位 | 核心职责 |
|------|------|----------|
| OpenCode | AI 编码平台 | 执行环境 |
| OpenSpec | 规范驱动开发 (SDD) | 变更管理、需求沉淀 |
| OpenPencil | AI 原生设计工具 | UI 设计稿生成 |

---

## 二、实施步骤

### 步骤 1：安装 OpenPencil CLI

```bash
npm install -g @zseven-w/openpencil
```

验证安装：

```bash
op --version
```

### 步骤 2：更新 opencode.json 配置 MCP 服务器

在 `opencode.json` 中添加 MCP 服务器配置：

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": [
    "superpowers@git+https://github.com/obra/superpowers.git"
  ],
  "mcp": {
    "openpencil": {
      "type": "remote",
      "url": "http://127.0.0.1:3100/mcp"
    }
  },
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

### 步骤 3：扩展 superpowers-bridge schema

在 `openspec/schemas/superpowers-bridge/schema.yaml` 中添加 `design-ui` artifact，
位于 specs → design → design-ui → tasks 流程中：

```yaml
- id: design-ui
  generates: design-ui/
  description: UI 设计稿（OpenPencil 格式），从 design.md 前端章节生成
  template: design-ui.md
  instruction: |
    使用 OpenPencil 生成 UI 设计稿。

    从 design.md 的 §Frontend Architecture 和 §UI Design Tokens 提取信息。
    为 specs/ 中的每个 capability 生成对应的 .op 设计文件。
    使用 MCP 工具或 op CLI 创建设计。

    输出目录：design-ui/<capability-name>.op
  requires:
    - design
```

### 步骤 4：创建 design-ui.md 模板

在 `openspec/schemas/superpowers-bridge/templates/` 中创建：

```markdown
# UI 设计稿

## 概述
本目录包含变更相关的 UI 设计稿，使用 OpenPencil 格式（.op 文件）。

## 关联 Capabilities

| Capability | 设计文件 | 说明 |
|------------|----------|------|
| {capability-name} | {capability-name}.op | {description} |

## 设计规范

从 design.md 提取的样式要求：

### 配色方案
- 主色：{primary-color}
- 辅助色：{secondary-color}
- 背景色：{background-color}

### 组件规范
- 按钮样式
- 卡片样式
- 列表样式

## 组件清单
- Header：顶部导航栏
- Content：主内容区域
- Footer：底部信息区

## 使用说明
1. 使用 OpenPencil 打开 .op 文件
2. 参考设计稿进行代码实现
3. 可导出为 PNG 或代码（React/Vue）
```

### 步骤 5：更新 tasks.md 模板

在 tasks 模板中添加 design-ui 引用格式：

```markdown
## 实现任务

### {模块名称}

- [ ] {task-id} {任务描述}
  - 设计稿: [design-ui/{capability-name}.op](design-ui/{capability-name}.op)
  - 技术栈: {tech-stack}
  - 验收标准: {acceptance-criteria}
```

### 步骤 6：更新 AGENTS.md

在 AGENTS.md 中添加 OpenPencil 协同规则：

```markdown
## OpenPencil 协同规则

使用 OpenPencil 作为 AI 原生设计工具，遵循 **需求(specs) → 架构(design) → UI(design-ui) → 前后端(tasks)** 的最佳实践。

### 集成阶段
- **specs（需求）阶段**：定义功能需求，包含前后端 Requirement 分类
- **design（架构）阶段**：在 §Frontend Architecture 和 §UI Design Tokens 章节确定前端架构和设计令牌 → 这是 design-ui 的**标准输入源**
- **design-ui（UI设计）阶段**：从 design.md 的 §Frontend Architecture 和 §UI Design Tokens 提取信息，生成 .op 设计文件
- **tasks（任务）阶段**：前后端任务分组，前端任务引用 .op 设计稿

### 设计文件位置
- `openspec/changes/<change-name>/design-ui/`

### 设计文件格式
- .op 文件（OpenPencil 原生格式）
- 可导出为 PNG、React、Vue 等

### MCP 工具
- `openpencil_design`：生成 UI 设计
- `openpencil_export`：导出设计稿

### 使用示例
1. design 阶段完成后，从 design.md 的前端章节提取设计令牌
2. AI 自动生成 design-ui 目录和 .op 文件
3. 在 tasks.md 中引用设计稿路径
4. 实现时使用 OpenPencil 打开参考
```

---

## 三、目录结构

```
<project-root>/
├── opencode.json                    # OpenCode 配置（含 MCP）
├── AGENTS.md                        # 项目指令（含 OpenPencil 规则）
├── openspec/
│   ├── config.yaml                  # OpenSpec 配置
│   ├── schemas/
│   │   └── superpowers-bridge/      # 扩展后的 schema
│   │       ├── schema.yaml          # 含 design-ui artifact
│   │       └── templates/
│   │           ├── design-ui.md     # 新增模板
│   │           └── ...
│   ├── changes/                     # 变更目录
│   └── specs/                       # 主 specs
└── doc/
    └── OpenCode-OpenSpec-OpenPencil-集成方案.md
```

---

## 四、使用示例

### 1. 创建变更

```bash
/opsx:new <feature-name> --schema superpowers-bridge
```

### 2. 自动生成文档

```bash
/opsx:ff
```

生成的文件（需求 → 架构 → UI → 前后端）：
- `brainstorm.md`
- `proposal.md`
- `specs/<capability-name>/spec.md` ← 需求（功能定义）
- `design.md` ← 架构（含前端架构 + 设计令牌）
- `design-ui/<capability-name>.op` ← UI（从 design.md 提取设计令牌）

### 3. 查看设计稿

```
openspec/changes/<change-name>/design-ui/
├── <capability-1>.op
├── <capability-2>.op
└── ...
```

### 4. 在任务中引用（前后端分组）

```markdown
## 1. 后端任务
- [ ] 1.1 开发后端逻辑
  - 规格: [specs/<capability-name>/spec.md](specs/<capability-name>/spec.md)

## 2. 前端任务
- [ ] 2.1 开发前端页面
  - 设计稿: [design-ui/<capability-name>.op](design-ui/<capability-name>.op)
  - 技术栈: {tech-stack}（从 design.md §Frontend Architecture 提取）
  - 验收标准: 与设计稿一致
  - 后端 API: [specs/<capability-name>/spec.md](specs/<capability-name>/spec.md)
```

### 5. 实现阶段

- 使用 OpenPencil 打开 .op 文件
- 参考设计稿编写代码
- 可导出为 PNG 进行视觉对比

---

## 五、验证清单

| 检查项 | 验证方式 |
|--------|----------|
| op CLI 安装 | `op --version` |
| MCP 服务器启动 | 启动 OpenPencil 桌面应用 |
| design-ui 目录生成 | 检查 `openspec/changes/<change-name>/design-ui/` |
| .op 文件有效性 | 使用 OpenPencil 打开 |
| tasks 引用正确 | 链接可打开 |

---

## 六、官方资源

- OpenPencil 官网：https://op.zseven.tech/
- OpenPencil GitHub：https://github.com/ZSeven-W/openpencil
- OpenPencil Skill：https://github.com/ZSeven-W/openpencil-skill
- MCP 工具文档：https://github.com/ZSeven-W/openpencil/tree/main/packages/pen-mcp
