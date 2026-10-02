/* Neural outputs select the pose; the garden supplies illustrative geometry. */
(function (root) {
  'use strict';
  const channels = ['proboscis', 'escape', 'grooming', 'walk_forward', 'walk_backward',
    'turn_left', 'turn_right', 'head', 'arousal'];
  function apiUrl(path) { return (root.FLYPET_API_BASE || '').replace(/\/$/, '') + path; }
  let session = '';
  function headers() { return {'Content-Type':'application/json', ...(session?{'X-Flypet-Session':session}:{})}; }
  async function connect() {
    if(session) {
      const check=await fetch(apiUrl('/api/garden/options'),{headers:headers(),signal:AbortSignal.timeout(15000)});
      if(check.ok)return;
      if(check.status!==401)throw new Error('The brain is busy. Please reconnect shortly.');
    }
    const response=await fetch(apiUrl('/api/garden/session'),{method:'POST',signal:AbortSignal.timeout(15000)});
    if(!response.ok)throw new Error('Could not start a visitor session. Try again shortly.');
    session=(await response.json()).session_id;
  }

  function interpret(result) {
    if (result?.source !== 'brian2' || !result.behaviour) throw new Error('Missing brain response.');
    const b = result.behaviour;
    if (channels.some(k => !Number.isFinite(b[k]) || b[k] < 0 || b[k] > 1)) {
      throw new Error('Invalid neural readout.');
    }
    // These are display thresholds, not claims about biological decision rules.
    if (b.escape > 0.1) return {mode: 'escape', strength: b.escape};
    if (b.proboscis > 0.1) return {mode: 'eat', strength: b.proboscis};
    if (b.grooming > 0.1) return {mode: 'groom', strength: b.grooming};
    const v = result.valence?.score;
    if (Number.isFinite(v) && Math.abs(v) > 0.1) {
      return {mode: v > 0 ? 'approach' : 'avoid', strength: Math.min(1, Math.abs(v))};
    }
    if (Math.max(b.walk_forward, b.walk_backward) > 0.1) {
      return {mode: 'walk', strength: Math.max(b.walk_forward, b.walk_backward),
        direction: b.walk_forward >= b.walk_backward ? 1 : -1};
    }
    if (Math.max(b.turn_left, b.turn_right) > 0.1) {
      return {mode: 'turn', strength: Math.max(b.turn_left, b.turn_right),
        direction: b.turn_right >= b.turn_left ? 1 : -1};
    }
    return {mode: 'idle', strength: 0};
  }

  function createClient({fetcher = (...args) => fetch(...args), onStatus = () => {}, onResult = () => {}} = {}) {
    let busy = false;
    async function send(input, context, reset = false) {
      if (busy) return false;
      busy = true;
      onStatus('busy');
      try {
        const response = await fetcher(apiUrl(reset ? '/api/garden/reset' : '/api/garden'), {
          method: 'POST', headers: headers(),
          body: JSON.stringify(input), signal: AbortSignal.timeout(180000),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(typeof result.error === 'string' ? result.error : `Request failed (${response.status}).`);
        if (reset && result.ok !== true) throw new Error('Invalid reset response.');
        if (result.source === 'dialogue' && typeof result.thought?.text !== 'string') throw new Error('Invalid dialogue response.');
        const decision = reset || result.source === 'dialogue' ? null : interpret(result);
        onResult(result, decision, context);
        onStatus('ready');
        return true;
      } catch (error) {
        onStatus('error', error.name === 'TimeoutError' ? 'Still waiting for the brain. Try again shortly.' : error.message);
        return false;
      } finally {
        busy = false;
        onStatus('settled');
      }
    }
    return {send: (input, context) => send(input, context), reset: input => send(input, {reset:true}, true), get busy() { return busy; }};
  }

  const api = {interpret, createClient, apiUrl, headers, connect};
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.GardenBrain = api;
})(globalThis);
