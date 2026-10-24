import katex from 'katex';
import 'katex/dist/katex.min.css';
export default function MathFormula({tex}:{tex:string}){const html=katex.renderToString(tex,{displayMode:true,throwOnError:false,output:'htmlAndMathml',strict:'warn'});return <div className="math-formula" dangerouslySetInnerHTML={{__html:html}}/>;}
