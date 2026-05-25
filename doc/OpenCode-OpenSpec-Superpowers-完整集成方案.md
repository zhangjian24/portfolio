# OpenCode + OpenSpec + Superpowers 完整集成方案

## 一、方案概述

使用 **superpowers-bridge** 官方 schema 将 Superpowers 深度集成到 OpenSpec 生命周期：

```
OpenSpec 规划层 → Superpowers 执行层
     ↓                    ↓
 proposal/design   →  superpowers:subagent-driven-development
                    →  TDD + code-review
```

### 三者定位

| 工具 | 定位 | 核心职责 |
|------|------|----------|
| OpenCode | AI 编码平台 | 执行环境 |
| OpenSpec | 规范驱动开发 (SDD) | 变更管理、需求沉淀 |
| Superpowers | 工程方法论 + 技能系统 | 执行质量、TDD、代码审查 |

---

## 二、实施步骤

### 步骤 1：更新 opencode.json

添加 Superpowers 插件：

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": [
    "superpowers@git+https://github.com/obra/superpowers.git"
  ],
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

### 步骤 2：下载 superpowers-bridge schema

```bash
# 克隆 schema 仓库
cd <your-project-directory>
git clone https://github.com/JiangWay/openspec-schemas.git /tmp/openspec-schemas

# 复制到项目
cp -r /tmp/openspec-schemas/superpowers-bridge openspec/schemas/
```

### 步骤 3：更新 AGENTS.md

将以下模板写入 `AGENTS.md`，替换其中 `{占位符}` 为项目实际信息：

```markdown
# {项目名称} - 项目指南

## 语言要求
- 所有思考过程和输出请用中文
- 使用简体中文回复
- 保持回复简洁，除非用户要求详细说明

## 技术栈
- {框架/语言/版本，如：Spring Boot 3.3.0 + MyBatis-Plus 3.5.7 + JDK 17}
- {数据库/中间件，如：MySQL + Redis}

## 项目结构
- {项目模块/层级概览}

## 核心业务模块
1. {业务模块一}
2. {业务模块二}
3. {业务模块三}

## 工作流配置

### OpenSpec + Superpowers 协同规则

使用 **superpowers-bridge** schema 进行深度集成：

1. **规划阶段** → 使用 OpenSpec + Superpowers brainstorming
   - 创建变更时指定 schema：`/opsx:new 任务名 --schema superpowers-bridge`
   - 第一阶段自动调用 `superpowers:brainstorming`

2. **实现阶段** → 使用 Superpowers TDD
   - `/opsx:apply` 自动调用 `superpowers:subagent-driven-development`
   - 每个任务强制 TDD：RED → GREEN → REFACTOR
   - 代码审查：两阶段 subagent 审查

3. **验证阶段** → 自定义 verify artifact
   - 5 项检查：结构验证、任务完成、Delta Spec 同步、设计/规格一致性、实现信号

### 开发规范
- 使用 Given/When/Then 格式描述测试场景
- 变更必须包含回滚方案
- 标注影响的模块和数据库表
- 输出语言：中文

## OpenSpec 命令（superpowers-bridge schema）

| 命令 | 说明 |
|------|------|
| /opsx:new 任务 --schema superpowers-bridge | 创建变更（使用 bridge schema） |
| /opsx:ff | 快速生成所有文档 |
| /opsx:apply | 执行（自动启用 TDD + code-review） |
| /opsx:verify | 验证 |
| /opsx:archive | 归档 |

## Superpowers 核心技能

| 技能 | 说明 |
|------|------|
| superpowers:brainstorming | 需求探索 |
| superpowers:writing-plans | 任务拆解为微步骤 |
| superpowers:subagent-driven-development | 子代理驱动开发 + TDD |
| superpowers:using-git-worktrees | Git worktree 隔离 |
| superpowers:finishing-a-development-branch | 分支收尾 |
| superpowers:requesting-code-review | 代码审查 |
```

---

## 三、目录结构

```
your-project/
├── opencode.json                    # OpenCode 配置（含 Superpowers 插件）
├── AGENTS.md                        # 项目指令（含协同规则）
├── openspec/
│   ├── config.yaml                  # OpenSpec 配置（默认 schema）
│   ├── schemas/
│   │   └── superpowers-bridge/      # 官方 bridge schema
│   │       ├── schema.yaml
│   │       ├── INTEGRATION.md
│   │       └── templates/
│   ├── changes/                     # 变更目录
│   └── specs/
└── doc/
```

---

## 四、使用示例

```bash
# 1. 创建变更（使用 superpowers-bridge schema）
/opsx:new 实现用户登录功能 --schema superpowers-bridge

# 2. 自动触发 brainstorming
# AI 会先调用 superpowers:brainstorming 探索需求

# 3. 生成规划文档
/opsx:ff

# 4. 执行（自动启用 TDD + subagent code-review）
/opsx:apply
# - 创建 git worktree
# - 每个任务强制 TDD 循环
# - 两阶段代码审查

# 5. 验证
/opsx:verify
# - 结构验证
# - 任务完成检查
# - Delta Spec 同步
# - 设计/规格一致性

# 6. 归档
/opsx:archive
```

---

## 五、检查清单

| 步骤 | 任务 | 状态 |
|------|------|------|
| 1 | 更新 opencode.json 添加 plugin | 待执行 |
| 2 | 重启 OpenCode 使插件生效 | 待执行 |
| 3 | 下载 superpowers-bridge schema | 待执行 |
| 4 | 更新 AGENTS.md（按通用模板填写项目信息） | 待执行 |
| 5 | 更新 openspec/config.yaml | 待执行 |
| 6 | 验证 schema 可用 | 待执行 |
| 7 | 测试创建变更 | 待执行 |

---

## 六、官方资源

- OpenSpec 官方文档：https://opencode.ai/docs/
- Superpowers 官方仓库：https://github.com/obra/superpowers
- superpowers-bridge schema：https://github.com/JiangWay/openspec-schemas/tree/main/superpowers-bridge
- OpenSpec 社区 schemas：https://github.com/Fission-AI/OpenSpec/blob/main/docs/customization.md
