import {describe,it,expect} from 'vitest';
import {initNetwork,forward,trainEpoch,metrics,dataset} from './nn';
describe('user-configurable hidden layers',()=>{
 it('constructs shape 2 → 5 → 4 → 3 → 1',()=>{
  const net=initNetwork([5,4,3],42);
  expect(net.layers).toEqual([2,5,4,3,1]);
  expect(net.weights.map(layer=>layer.map(row=>row.length))).toEqual([
   Array(5).fill(2),Array(4).fill(5),Array(3).fill(4),[3]
  ]);
  expect(net.biases.map(layer=>layer.length)).toEqual([5,4,3,1]);
 });
 it('supports one through four hidden layers during forward and backward passes',()=>{
  const data=dataset('xor',24,4);
  for(const layers of [[2],[4,3],[6,5,4],[4,3,2,2]]){
   const net=initNetwork(layers,3),next=trainEpoch(net,data,.2,'tanh');
   expect(next.layers).toEqual([2,...layers,1]);
   expect(Number.isFinite(metrics(next,data,'tanh').loss)).toBe(true);
   expect(forward(next,[.2,-.3]).probability).toBeGreaterThan(0);
   expect(forward(next,[.2,-.3]).probability).toBeLessThan(1);
  }
 });
 it('updates the early layer of a deeper network through backpropagation',()=>{
  const net=initNetwork([5,4,3],9),trained=trainEpoch(net,dataset('linear',60,7),.5,'tanh');
  expect(trained.weights[0]).not.toEqual(net.weights[0]);
  expect(trained.weights[3]).not.toEqual(net.weights[3]);
  expect(net.weights).toEqual(initNetwork([5,4,3],9).weights);
 });
 it('keeps seeded initialization reproducible for multiple layers',()=>{
  expect(initNetwork([8,4,2],42)).toEqual(initNetwork([8,4,2],42));
 });
});
