export type Sample={x:number;y:number;label:0|1};
export type Dataset='xor'|'circle'|'linear'|'spiral';
export type Network={layers:number[];weights:number[][][];biases:number[][]};
export type Activation='tanh'|'relu';
export function sigmoid(v:number){return v>=0?1/(1+Math.exp(-v)):Math.exp(v)/(1+Math.exp(v))}
export function activate(v:number,type:Activation){return type==='tanh'?Math.tanh(v):Math.max(0,v)}
export function derivative(z:number,type:Activation){return type==='tanh'?1-Math.tanh(z)**2:z>0?1:0}
export function seeded(seed:number){let x=seed>>>0;return()=>{x=(1664525*x+1013904223)>>>0;return x/4294967296}}
export function dataset(type:Dataset,count=160,seed=7):Sample[]{const random=seeded(seed);const out:Sample[]=[];
for(let i=0;i<count;i++){const label=(i%2) as 0|1;let x=0,y=0;
if(type==='xor'){x=(label===0?1:-1)*(random()>.5?1:-1)*(.3+.7*random());y=(label===0?1:-1)*Math.sign(x)*(.3+.7*random())}
else if(type==='circle'){const angle=random()*Math.PI*2,r=label===1?.1+.45*random():.65+.3*random();x=r*Math.cos(angle);y=r*Math.sin(angle)}
else if(type==='linear'){x=random()*2-1;y=(label===1?.35:-.35)+.6*x+(random()-.5)*.45}
else{const r=.1+.85*random();const angle=r*3*Math.PI+(label*Math.PI)+(random()-.5)*.35;x=r*Math.cos(angle);y=r*Math.sin(angle)}
out.push({x,y,label});}return out}
export function initNetwork(hidden:number[],seed=12):Network{const layers=[2,...hidden,1],rnd=seeded(seed);const weights:number[][][]=[],biases:number[][]=[];
for(let l=1;l<layers.length;l++){const scale=Math.sqrt(2/layers[l-1]);weights.push(Array.from({length:layers[l]},()=>Array.from({length:layers[l-1]},()=>((rnd()+rnd()+rnd()+rnd())-2)*scale)));biases.push(Array(layers[l]).fill(0));}
return{layers,weights,biases}}
export function forward(net:Network,input:[number,number],activation:Activation='tanh'){const a:number[][]=[input],z:number[][]=[];for(let l=0;l<net.weights.length;l++){const zz=net.weights[l].map((row,j)=>row.reduce((sum,w,k)=>sum+w*a[l][k],net.biases[l][j]));z.push(zz);a.push(zz.map(v=>l===net.weights.length-1?sigmoid(v):activate(v,activation)));}return{a,z,probability:a[a.length-1][0]}}
export function metrics(net:Network,data:Sample[],activation:Activation){let loss=0,correct=0;for(const s of data){const p=forward(net,[s.x,s.y],activation).probability;loss+=-(s.label*Math.log(Math.max(1e-12,p))+(1-s.label)*Math.log(Math.max(1e-12,1-p)));correct+=(p>=.5?1:0)===s.label?1:0;}return{loss:loss/data.length,accuracy:correct/data.length}}
export function trainEpoch(net:Network,data:Sample[],rate:number,activation:Activation):Network{
const weights=net.weights.map(layer=>layer.map(row=>row.slice())),biases=net.biases.map(b=>b.slice());const gw=weights.map(layer=>layer.map(row=>row.map(()=>0))),gb=biases.map(row=>row.map(()=>0));
for(const sample of data){const{a,z,probability}=forward(net,[sample.x,sample.y],activation);let delta=[probability-sample.label];for(let l=weights.length-1;l>=0;l--){for(let j=0;j<delta.length;j++){gb[l][j]+=delta[j];for(let k=0;k<a[l].length;k++)gw[l][j][k]+=delta[j]*a[l][k];}if(l>0){const prev=Array.from({length:a[l].length},(_,k)=>weights[l].reduce((sum,row,j)=>sum+row[k]*delta[j],0)*derivative(z[l-1][k],activation));delta=prev;}}}
for(let l=0;l<weights.length;l++)for(let j=0;j<weights[l].length;j++){biases[l][j]-=rate*gb[l][j]/data.length;for(let k=0;k<weights[l][j].length;k++)weights[l][j][k]-=rate*gw[l][j][k]/data.length;}return{layers:net.layers,weights,biases}}
