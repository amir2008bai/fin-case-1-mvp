const { chromium } = require('playwright');const fs=require('fs');
(async()=>{const b=await chromium.launch();const p=await b.newPage({viewport:{width:1920,height:1080}});
await p.goto('file://'+__dirname+'/index.html');await p.waitForTimeout(300);
const [a,z,step]=process.argv.slice(2).map(Number);fs.mkdirSync(__dirname+'/f',{recursive:true});
for(let i=a;i<z;i+=step){await p.evaluate(t=>render(t),i/60);await p.screenshot({path:`${__dirname}/f/${String(i).padStart(5,'0')}.jpg`,type:'jpeg',quality:92});}
await b.close();})();
