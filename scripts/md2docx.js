import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType, BorderStyle, AlignmentType } from 'docx';
import { marked } from 'marked';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function parseInline(text) {
  const runs = [];
  let remaining = text;

  while (remaining.length > 0) {
    let match = remaining.match(/\*\*\*(.+?)\*\*\*/);
    if (match) {
      const idx = remaining.indexOf(match[0]);
      if (idx > 0) runs.push(new TextRun({ text: remaining.slice(0, idx), font: 'SimSun' }));
      runs.push(new TextRun({ text: match[1], bold: true, italics: true, font: 'SimSun' }));
      remaining = remaining.slice(idx + match[0].length);
      continue;
    }
    match = remaining.match(/\*\*(.+?)\*\*/);
    if (match) {
      const idx = remaining.indexOf(match[0]);
      if (idx > 0) runs.push(new TextRun({ text: remaining.slice(0, idx), font: 'SimSun' }));
      runs.push(new TextRun({ text: match[1], bold: true, font: 'SimSun' }));
      remaining = remaining.slice(idx + match[0].length);
      continue;
    }
    match = remaining.match(/\*(.+?)\*/);
    if (match) {
      const idx = remaining.indexOf(match[0]);
      if (idx > 0) runs.push(new TextRun({ text: remaining.slice(0, idx), font: 'SimSun' }));
      runs.push(new TextRun({ text: match[1], italics: true, font: 'SimSun' }));
      remaining = remaining.slice(idx + match[0].length);
      continue;
    }
    match = remaining.match(/`(.+?)`/);
    if (match) {
      const idx = remaining.indexOf(match[0]);
      if (idx > 0) runs.push(new TextRun({ text: remaining.slice(0, idx), font: 'SimSun' }));
      runs.push(new TextRun({ text: match[1], font: 'Courier New', size: 18, shading: { type: 'clear', color: 'auto', fill: 'F3F3F3' } }));
      remaining = remaining.slice(idx + match[0].length);
      continue;
    }
    match = remaining.match(/\[(.+?)\]\((.+?)\)/);
    if (match) {
      const idx = remaining.indexOf(match[0]);
      if (idx > 0) runs.push(new TextRun({ text: remaining.slice(0, idx), font: 'SimSun' }));
      runs.push(new TextRun({ text: match[1], color: '0563C1', underline: {} }));
      remaining = remaining.slice(idx + match[0].length);
      continue;
    }
    runs.push(new TextRun({ text: remaining, font: 'SimSun' }));
    break;
  }
  return runs;
}

function parseCodeBlock(code) {
  const lines = code.split('\n');
  return lines.map(line => new Paragraph({
    children: [new TextRun({ text: line, font: 'Courier New', size: 18 })],
    style: 'CodeBlock',
    spacing: { before: 0, after: 0 },
  }));
}

function parseTableRow(rowText, isHeader) {
  const cells = rowText.split('|').filter((_, i, arr) => i > 0 && i < arr.length - 1).map(cell => {
    return new TableCell({
      children: [new Paragraph({
        children: parseInline(cell.trim()),
        alignment: AlignmentType.LEFT,
      })],
      shading: isHeader ? { type: 'clear', color: 'auto', fill: 'D9E2F3' } : undefined,
    });
  });
  return new TableRow({ children: cells });
}

function convertMarkdownToDocx(inputPath, outputPath) {
  const markdown = fs.readFileSync(inputPath, 'utf-8');
  const content = markdown.replace(/^---[\s\S]*?---\n/, '');
  const children = [];
  const lines = content.split('\n');
  let i = 0;
  let inCodeBlock = false;
  let codeBlockContent = '';
  let blockquoteContent = [];

  while (i < lines.length) {
    const line = lines[i];

    if (line.startsWith('```')) {
      if (!inCodeBlock) {
        inCodeBlock = true;
        codeBlockContent = '';
      } else {
        inCodeBlock = false;
        children.push(...parseCodeBlock(codeBlockContent.trimEnd()));
        children.push(new Paragraph({ children: [] }));
      }
      i++;
      continue;
    }

    if (inCodeBlock) {
      codeBlockContent += line + '\n';
      i++;
      continue;
    }

    if (line.startsWith('>')) {
      blockquoteContent.push(line.slice(1).trim());
      i++;
      continue;
    } else if (blockquoteContent.length > 0) {
      children.push(new Paragraph({
        children: [new TextRun({ text: blockquoteContent.join(' '), italics: true, color: '666666', font: 'SimSun' })],
        border: { left: { color: 'CCCCCC', space: 4, style: BorderStyle.SINGLE, size: 6 } },
        indent: { left: 720 },
      }));
      blockquoteContent = [];
      continue;
    }

    if (line.startsWith('# ')) {
      children.push(new Paragraph({
        children: [new TextRun({ text: line.slice(2), font: 'SimHei', size: 40, bold: true })],
        heading: HeadingLevel.HEADING_1,
        spacing: { before: 400, after: 200 },
      }));
      i++;
      continue;
    }

    if (line.startsWith('## ')) {
      children.push(new Paragraph({
        children: [new TextRun({ text: line.slice(3), font: 'SimHei', size: 32, bold: true })],
        heading: HeadingLevel.HEADING_2,
        spacing: { before: 300, after: 150 },
      }));
      i++;
      continue;
    }

    if (line.startsWith('### ')) {
      children.push(new Paragraph({
        children: [new TextRun({ text: line.slice(4), font: 'SimHei', size: 28, bold: true })],
        heading: HeadingLevel.HEADING_3,
        spacing: { before: 240, after: 120 },
      }));
      i++;
      continue;
    }

    if (line.startsWith('#### ')) {
      children.push(new Paragraph({
        children: [new TextRun({ text: line.slice(5), font: 'SimHei', size: 24, bold: true })],
        heading: HeadingLevel.HEADING_4,
        spacing: { before: 200, after: 100 },
      }));
      i++;
      continue;
    }

    if (line.match(/^---+$/)) {
      children.push(new Paragraph({
        children: [],
        border: { bottom: { color: 'CCCCCC', space: 1, style: BorderStyle.SINGLE, size: 6 } },
        spacing: { before: 200, after: 200 },
      }));
      i++;
      continue;
    }

    if (line.includes('|') && line.trim().startsWith('|')) {
      const tableRows = [];
      while (i < lines.length && lines[i].includes('|')) {
        const rowText = lines[i].trim();
        if (!rowText.match(/^[\|\s\-:]+$/)) {
          tableRows.push(rowText);
        }
        i++;
      }
      if (tableRows.length > 0) {
        const isFirstRowHeader = true;
        const table = new Table({
          rows: tableRows.map((rowText, idx) => parseTableRow(rowText, idx === 0 && isFirstRowHeader)),
          width: { size: 100, type: WidthType.PERCENTAGE },
        });
        children.push(table);
        children.push(new Paragraph({ children: [] }));
      }
      continue;
    }

    if (line.match(/^[\-\*] /)) {
      const text = line.replace(/^[\-\*] /, '');
      children.push(new Paragraph({
        children: [
          new TextRun({ text: '•  ', font: 'SimSun' }),
          ...parseInline(text),
        ],
        indent: { left: 360 },
        spacing: { before: 60, after: 60 },
      }));
      i++;
      continue;
    }

    const numberedMatch = line.match(/^(\d+)\. (.+)/);
    if (numberedMatch) {
      children.push(new Paragraph({
        children: [
          new TextRun({ text: numberedMatch[1] + '.  ', font: 'SimSun' }),
          ...parseInline(numberedMatch[2]),
        ],
        indent: { left: 360 },
        spacing: { before: 60, after: 60 },
      }));
      i++;
      continue;
    }

    if (line.trim() === '') {
      children.push(new Paragraph({ children: [] }));
      i++;
      continue;
    }

    children.push(new Paragraph({
      children: parseInline(line),
      spacing: { before: 60, after: 60 },
    }));

    i++;
  }

  const doc = new Document({
    styles: {
      paragraphStyles: [
        {
          id: 'CodeBlock',
          name: 'Code Block',
          run: { font: 'Courier New', size: 18 },
          paragraph: {
            spacing: { before: 0, after: 0 },
            shading: { type: 'clear', color: 'auto', fill: 'F5F5F5' },
          },
        },
      ],
    },
    sections: [{
      properties: {
        page: {
          margin: {
            top: 1440,
            right: 1440,
            bottom: 1440,
            left: 1440,
          },
        },
      },
      children: children,
    }],
  });

  Packer.toBuffer(doc).then((buffer) => {
    fs.writeFileSync(outputPath, buffer);
    console.log(`Generated: ${outputPath}`);
  });
}

const args = process.argv.slice(2);
if (args.length < 1) {
  console.log('Usage: node scripts/md2docx.js <input.md> [output.docx]');
  console.log('Example: node scripts/md2docx.js src/content/blog/article.md');
  console.log('         node scripts/md2docx.js src/content/blog/article.md output.docx');
  process.exit(1);
}

const inputPath = path.resolve(args[0]);
const outputPath = args[1] 
  ? path.resolve(args[1]) 
  : inputPath.replace(/\.mdx?$/, '.docx');

convertMarkdownToDocx(inputPath, outputPath);
