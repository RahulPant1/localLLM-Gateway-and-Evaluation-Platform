import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/** Model output rendered as Markdown (headings, lists, code blocks, tables). Raw HTML is not rendered. */
export function Markdown({ text }: { text: string }) {
  return (
    <div className="markdown">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
    </div>
  );
}
