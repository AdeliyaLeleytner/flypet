/* Full controls remain independent of canvas artwork and never insert model HTML. */
(function(root) {
  'use strict';
  function attach({client,onError,contextForFruit,canSend}) {
    const $ = id => document.getElementById(id);
    let options=null, generation=0, lastTrial=null, activeThought=null;
    const names={apple:'apple',pear:'pear',plum:'plum',melon:'melon',mushroom:'mushroom',grape:'grape'};
    function readNumber(id,min,max,optional=false,integer=false) {
      const raw=$(id).value.trim();
      if(optional&&!raw)return null;
      const value=Number(raw);
      if(!raw||!Number.isFinite(value)||value<min||value>max||(integer&&!Number.isInteger(value))) {
        throw new Error('Check “'+($(id).closest('label')?.firstChild?.textContent?.trim()||id)+'”: expected '+min+'–'+max+'.');
      }
      return value;
    }
    function enrich(input) {
      const req={...input,side:$('trial-side').value,read_brain:$('trial-reader').checked};
      const duration=readNumber('trial-duration',100,1000,true,true),seed=readNumber('trial-seed',0,4294967295,true,true);
      const rate=readNumber('trial-rate',0,300,true);
      if(duration!==null)req.duration_ms=duration;
      if(seed!==null)req.seed=seed;
      if(rate!==null&&['sugar','bitter','touch','looming'].includes(req.action))req.rate_hz=rate;
      return req;
    }
    function send(input,context={}) {
      if(!canSend()||client.busy)return;
      try{return client.send(enrich(input),context)}catch(error){onError(error.message)}
    }
    function state(value) {
      if(!value)return;
      $('pet-state').textContent='Satiety: '+Math.round(value.satiety*100)+'% · trials: '+value.n_stimuli;
      $('memory-status').textContent='Memory strength: '+(100*value.memory_strength).toFixed(2)+'%';
      $('memory-associations').textContent=(value.associations||[]).join('\n')||'No learned associations yet.';
    }
    function showThought(thought) {
      if(!thought)return;
      $('fly-thought').textContent=thought.text;
      $('thought-source').textContent=thought.source==='llm'?'Language-model reflection · '+(thought.model||[]).join(', '):
        thought.status==='unavailable'?'Readout description · language model unavailable':
        thought.status==='busy'?'Readout description · language model busy':'Direct description of neural readouts';
    }
    async function languageThought(trial,version) {
      if(!trial||activeThought===trial)return;
      activeThought=trial;$('retry-thought').disabled=true;
      $('thought-source').textContent='Response ready · composing a reflection…';
      try {
        const response=await fetch(GardenBrain.apiUrl('/api/garden/thought'),{
          method:'POST',headers:GardenBrain.headers(),body:JSON.stringify({trial_id:trial}),
          signal:AbortSignal.timeout(180000)});
        const result=await response.json();
        if(version!==generation||trial!==lastTrial)return;
        if(result.thought)showThought(result.thought);
        else throw new Error(result.error||'Response unavailable.');
      } catch(error) {
        if(version===generation&&trial===lastTrial)$('thought-source').textContent='Readout description retained · '+error.message;
      } finally {
        if(activeThought===trial)activeThought=null;
        if(version===generation)$('retry-thought').disabled=!lastTrial;
      }
    }
    function begin() {generation++;lastTrial=null;$('retry-thought').disabled=true;}
    function present(result) {
      state(result.state);
      if(result.reset) {
        lastTrial=null;
        $('fly-thought').textContent='My state has been updated. Offer me another stimulus.';
        $('thought-source').textContent='Reset: '+Object.entries(result.reset).filter(([,v])=>v).map(([k])=>k).join(', ');
        $('memory-learning').textContent='';
        for(const id of ['reader-output','vnc-output'])$(id).textContent='No trial since the reset.';
        return;
      }
      showThought(result.thought);
      if(result.source==='dialogue') {
        $('chat-status').textContent='Reply without a simulation.';
        return;
      }
      lastTrial=result.trial_id;
      $('retry-thought').disabled=!lastTrial;
      $('chat-status').textContent=result.action==='chat'?'Your words became stimuli; the simulated response is shown.':'You can describe the next stimulus in words.';
      $('reader-output').textContent=result.brain_reading||'Not requested for this trial.';
      $('vnc-output').textContent=result.body?JSON.stringify(result.body,null,2):'VNC is disabled on this server.';
      $('memory-learning').textContent=result.learning?
        'Reinforcement: '+result.learning.reinforcement+' · modified synapses: '+result.learning.synapses_changed:
        result.valence?'Naive: '+result.valence.naive_score+' · learned: '+result.valence.score+' · Δ '+result.valence.delta:'';
      if($('llm-thoughts').checked&&lastTrial)languageThought(lastTrial,generation);
    }
    function appendOption(select,value,label) {
      const option=document.createElement('option');option.value=value;option.textContent=label;select.append(option);
    }
    function channelFields() {
      const container=$('channel-fields');container.replaceChildren();
      const config=options.channels[$('physical-channel').value];
      for(const [key,spec] of Object.entries(config.fields)) {
        const label=document.createElement('label');label.append(document.createTextNode(spec.label));
        const control=document.createElement(spec.options?'select':'input');
        control.dataset.parameter=key;control.id='channel-'+key;
        if(spec.options)for(const value of spec.options)appendOption(control,value,value);
        else {control.type='number';control.min=spec.min;control.max=spec.max;control.step=spec.step;}
        control.value=spec.default;label.append(control);container.append(label);
      }
    }
    async function load() {
      await GardenBrain.connect();
      const response=await fetch(GardenBrain.apiUrl('/api/garden/options'),{headers:GardenBrain.headers(),signal:AbortSignal.timeout(15000)});
      if(!response.ok)throw new Error('Could not load the stimulus catalog.');
      options=await response.json();
      const catalog=$('stimulus-catalog');catalog.replaceChildren();
      for(const item of options.stimuli) {
        const label=document.createElement('label'),box=document.createElement('input');
        box.type='checkbox';box.value=item.key;box.dataset.rate=item.rate_hz;
        box.addEventListener('change',()=>{
          if(catalog.querySelectorAll('input:checked').length>4){box.checked=false;onError('Select at most four inputs.');}
        });
        label.append(box,document.createTextNode(' '+item.label));catalog.append(label);
      }
      $('physical-channel').replaceChildren();
      for(const [key,value] of Object.entries(options.channels))appendOption($('physical-channel'),key,value.label);
      channelFields();
      $('odor-compound').replaceChildren();
      for(const name of options.odors)appendOption($('odor-compound'),name,name);
      if(options.odors.includes('ethyl acetate'))$('odor-compound').value='ethyl acetate';
      const reader=options.components.reader?.status==='ready';
      $('trial-reader').disabled=!reader;if(!reader)$('trial-reader').checked=false;
      $('reader-availability').textContent=reader?'available':'disabled on this server';
      $('llm-thoughts').disabled=!options.llm_thoughts;
      if(!options.llm_thoughts)$('llm-thoughts').checked=false;
      state(options.state);
    }
    function mixture(keys) {
      const rate=readNumber('trial-rate',0,300,true);
      return {action:'stimuli',stimuli:keys.map(key=>rate===null?{key}:{key,rate_hz:rate})};
    }
    $('run-mixture').addEventListener('click',()=>{
      try {
        const keys=[...$('stimulus-catalog').querySelectorAll('input:checked')].map(x=>x.value);
        if(!keys.length)throw new Error('Select some stimuli first.');
        send(mixture(keys));
      }catch(error){onError(error.message)}
    });
    $('sugar-bitter').addEventListener('click',()=>{try{send(mixture(['sugar','bitter']))}catch(error){onError(error.message)}});
    $('physical-channel').addEventListener('change',channelFields);
    $('run-channel').addEventListener('click',()=>{
      try {
        const params={};
        for(const input of $('channel-fields').querySelectorAll('[data-parameter]'))params[input.dataset.parameter]=input.tagName==='SELECT'?input.value:readNumber(input.id,Number(input.min),Number(input.max));
        send({action:'channel',channel:$('physical-channel').value,params});
      }catch(error){onError(error.message)}
    });
    function odorVisibility() {
      const mode=$('odor-mode').value;
      $('fruit-field').hidden=mode!=='fruit';$('compound-field').hidden=mode!=='compound';$('odor-text-field').hidden=mode!=='text';
    }
    $('odor-mode').addEventListener('change',odorVisibility);odorVisibility();
    for(const [id,reinforcement] of [['odor-probe','none'],['odor-reward','reward'],['odor-punish','punish']]) {
      $(id).addEventListener('click',()=>{
        try {
          const req={action:'smell',reinforcement,odor_scale:readNumber('odor-scale',0,1)};
          const mode=$('odor-mode').value;
          if(mode==='fruit')req.target=$('odor-fruit').value;
          else if(mode==='compound')req.compound=$('odor-compound').value;
          else {req.odor_text=$('odor-object').value.trim();if(!req.odor_text)throw new Error('Enter an object.');}
          send(req,req.target?contextForFruit(req.target):{});
        }catch(error){onError(error.message)}
      });
    }
    $('garden-chat').addEventListener('submit',event=>{
      event.preventDefault();const text=$('garden-message').value.trim();
      if(text)send({action:'chat',text});
    });
    for(const [id,scope] of [['reset-body','body'],['reset-history','history'],['reset-memory','memory']]) {
      $(id).addEventListener('click',()=>{
        if(canSend()&&!client.busy)client.reset({body:scope==='body',memory:scope==='memory',history:scope==='history'});
      });
    }
    $('retry-thought').addEventListener('click',()=>languageThought(lastTrial,generation));
    return {enrich,present,begin,load,selectFruit:id=>{$('odor-fruit').value=id;}};
  }
  root.GardenControls={attach};
})(globalThis);
