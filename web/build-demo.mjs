// Build an isolated static demo from allowlisted interface files and synthetic examples.
import {mkdir,readFile,writeFile,copyFile,cp} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const root=path.dirname(fileURLToPath(import.meta.url));
const destination=path.resolve(process.argv[2]||path.join(root,'dist'));
if(destination===root||destination===path.dirname(root))throw new Error('Choose a separate output directory.');
await mkdir(destination,{recursive:true});
for(const file of ['app.js','domain.js','transport.js','styles.css'])await copyFile(path.join(root,'public',file),path.join(destination,file));
await cp(path.join(root,'public','assets'),path.join(destination,'assets'),{recursive:true});
let html=await readFile(path.join(root,'public','index.html'),'utf8');
html=html.replace('<title>CatalogIQ · Product workspace</title>','<title>CatalogIQ · Team B demo</title>');
// Static demo metadata; the local Python server also sends HTTP security headers.
html=html.replace('<head>','<head>\n  <meta name="catalogiq-runtime" content="static-demo">\n  <meta name="robots" content="noindex,nofollow">\n  <meta name="referrer" content="no-referrer">\n  <meta http-equiv="Content-Security-Policy" content="object-src \'none\'; base-uri \'none\'">');
await writeFile(path.join(destination,'index.html'),html);
const examples={};
for(const kind of ['request','response'])examples[kind]=JSON.parse(await readFile(path.join(root,'..','docs','integration',`inference_adapter_${kind}.json`),'utf8'));
if(examples.request.products.length!==2||examples.response.products.length!==2||examples.request.products.some(p=>!p.record_id.startsWith('demo-')))throw new Error('Only the two reviewed synthetic examples may be published.');
await writeFile(path.join(destination,'examples.json'),JSON.stringify(examples,null,2)+'\n');
await writeFile(path.join(destination,'robots.txt'),'User-agent: *\nDisallow: /\n');
console.log(JSON.stringify({directory:destination,examples:2,runtime:'browser-only demo'}));
