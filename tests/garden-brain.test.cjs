const test = require('node:test');
const assert = require('node:assert/strict');
const {interpret, createClient} = require('../flypet/static/garden-brain.js');
const empty = {proboscis:0, escape:0, grooming:0, walk_forward:0, walk_backward:0,
  turn_left:0, turn_right:0, head:0, arousal:0};
const result = (b={}, extra={}) => ({source:'brian2', behaviour:{...empty,...b}, ...extra});

test('zero output never animates the stimulus label; output drives pose', () => {
  for(const action of ['sugar','looming','touch','smell']) assert.equal(interpret(result({}, {action})).mode, 'idle');
  assert.equal(interpret(result({grooming:.7}, {action:'sugar'})).mode, 'groom');
  assert.equal(interpret(result({escape:.8,proboscis:1})).mode, 'escape');
  assert.equal(interpret(result({proboscis:.6})).mode, 'eat');
});
test('opposite odor valences reverse movement and malformed output is rejected', () => {
  assert.equal(interpret(result({}, {valence:{score:.4}})).mode,'approach');
  assert.equal(interpret(result({}, {valence:{score:-.4}})).mode,'avoid');
  assert.equal(interpret(result({walk_backward:.8})).direction,-1);
  assert.equal(interpret(result({turn_left:.8})).direction,-1);
  assert.throws(()=>interpret({source:'animation',behaviour:empty}));
  assert.throws(()=>interpret(result({escape:NaN})));
});
test('client suppresses concurrent events and only applies a completed brain response', async () => {
  let finish, calls=0, applied=[];
  const client=createClient({fetcher:()=>{calls++;return new Promise(resolve=>{finish=resolve})},
    onResult:(r,d,c)=>applied.push({d,c})});
  const first=client.send({action:'looming'},{x:12});
  assert.equal(client.busy,true);
  assert.equal(await client.send({action:'sugar'}),false);
  assert.equal(applied.length,0);
  finish({ok:true,json:async()=>result({escape:.7})});
  assert.equal(await first,true);
  assert.equal(calls,1);
  assert.deepEqual(applied,[{d:{mode:'escape',strength:.7},c:{x:12}}]);
  assert.equal(client.busy,false);
});
test('failed, busy, or invalid responses never trigger an animated fallback', async () => {
  for(const fetcher of [async()=>{throw new Error('offline')},
    async()=>({ok:false,status:429,json:async()=>({error:'busy'})}),
    async()=>({ok:true,json:async()=>({})})]){
    const statuses=[];let applied=0;
    const client=createClient({fetcher,onResult:()=>applied++,onStatus:s=>statuses.push(s)});
    assert.equal(await client.send({action:'looming'}),false);
    assert.equal(applied,0);
    assert.equal(client.busy,false);
    assert.ok(statuses.includes('error'));
  }
});
