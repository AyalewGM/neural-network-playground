import {describe,it,expect} from 'vitest';
import {initNetwork,dataset,forward,metrics,trainEpoch,derivative} from './nn';
describe('Neural Network Playground',()=>{
 it('initializes reproducibly',()=>expect(initNetwork([5],33)).toEqual(initNetwork([5],33)));
 it('generates balanced deterministic datasets',()=>{const a=dataset('xor',100,7);expect(a).toEqual(dataset('xor',100,7));expect(a.filter(s=>s.label===1)).toHaveLength(50)});
 it('predicts probabilities within zero and one',()=>{const n=initNetwork([4]);for(const x of [-1,0,1]){const p=forward(n,[x,-x]).probability;expect(p).toBeGreaterThan(0);expect(p).toBeLessThan(1)}});
 it('tanh derivative is one at zero',()=>expect(derivative(0,'tanh')).toBeCloseTo(1));
 it('learning lowers classification loss',()=>{const data=dataset('linear',120,11);let net=initNetwork([8],19);const before=metrics(net,data,'tanh').loss;for(let k=0;k<250;k++)net=trainEpoch(net,data,.8,'tanh');expect(metrics(net,data,'tanh').loss).toBeLessThan(before)});
 it('training never mutates the input network',()=>{const net=initNetwork([3],9);const snap=JSON.stringify(net);trainEpoch(net,dataset('xor',20),.25,'tanh');expect(JSON.stringify(net)).toBe(snap)});
});
