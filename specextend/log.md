
=== Condition: cloud_ar ===
Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]Loading checkpoint shards:  50%|█████     | 1/2 [00:00<00:00,  8.21it/s]Loading checkpoint shards: 100%|██████████| 2/2 [00:00<00:00,  9.35it/s]
/root/code/long-ecsd/specextend/.venv/lib/python3.12/site-packages/transformers/generation/configuration_utils.py:515: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.9` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`. This was detected when initializing the generation config instance, which means the corresponding file may hold incorrect parameterization and should be fixed.
  warnings.warn(
/root/code/long-ecsd/specextend/.venv/lib/python3.12/site-packages/transformers/generation/configuration_utils.py:520: UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.6` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`. This was detected when initializing the generation config instance, which means the corresponding file may hold incorrect parameterization and should be fixed.
  warnings.warn(
/root/code/long-ecsd/specextend/.venv/lib/python3.12/site-packages/transformers/generation/configuration_utils.py:515: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.9` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
  warnings.warn(
/root/code/long-ecsd/specextend/.venv/lib/python3.12/site-packages/transformers/generation/configuration_utils.py:520: UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.6` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`.
  warnings.warn(
Sp ain has a long Sp ain has a long Sp ain has a long 
=== Sample 1/3 [cloud_ar] ===
Sp ain has a long and complex history , marked by political inst ability , economic strugg les , and social un rest . After the death of dict ator Francisco Franco in  1 9 7 5 , Spain moved from dict ator ship to dem ocracy and began a period of rapid economic growth . In  1 9 8 2 , Spain joined NAT O and the European Union in  1 9 8 6 . Since then , Spain has played an increasing ly prominent role in international bodies , with Spanish diplom ats and polit icians often assuming positions of leadership . The United States has retained access to several Spanish military bases , including a naval base at R ota and an air base at Mor on , in exchange for economic aid to the Franco regime . Despite this , the United States and Spain have generally maintained good relations in the post - Fran co period , whether under Social ist or right - of - center govern ments . 
 
 The Social ist Party ( P SO E ), led by Prime Minister José Luis Rodríguez Z ap ater o , has govern ed Spain since  2 0 0 4 . Z ap ater o and the P SO E won a second term in office in March  2 0 0 8 parliament ary elections , with the P SO E winning  1 6 9 seats in the  3 5 0 - 
[cloud_ar] Generated 256 tokens in 11.61s. Token/sec: 22.05
Wrote metrics to edge_cloud_metrics_4K_3samples_cloud_ar_sample1.json

=== Sample 2/3 [cloud_ar] ===
The Federal Records Act requires federal ag encies to manage and preserve records that document their organization , functions , policies , dec isions , procedures , and essential transactions . E - mail records are considered federal records and must be effectively managed to avoid legal li abilities , loss of benefits , and loss of historical records . A gen cies must develop records management programs , identify and invent ory records , app raise their value , determine their perman ence , and issue management direct ives . The National Archives and Records Administration ( N AR A ) over se es and gu ides federal records management , including e - mail records . N AR A works with ag encies to schedule records and appro ves all records sched ules . E - mail records must be preserved with certain transmission data , and ag encies must establish policies and procedures for their ret ention and disposition . N AR A has issued reg ulations for the management of e - mail records , and ag encies may use either paper or electronic record keep ing systems for record copies of e - mail messages , depending on their business needs . The advantages of using an electronic record keep ing system include efficient searching and classification of records , but the dis adv antages include the risk of loss from in ad vert ent or automatic delet ion and the need for careful planning and analysis of ag ency requirements 
[cloud_ar] Generated 256 tokens in 10.45s. Token/sec: 24.5
Wrote metrics to edge_cloud_metrics_4K_3samples_cloud_ar_sample2.json

=== Sample 3/3 [cloud_ar] ===
B io fu els are alternative transport ation fu els derived from renew able resources . Currently , most bio fu els are produced from corn and so y beans , with eth an ol being the most common bio f uel in the United States . However , next generation bio fu els such as cell ulos ic eth an ol and alg ae - based fu els are being promoted for various reasons , including their potential to boost energy independence and less en environmental impact s . Cell ulos ic feed stock s include annual or per enn ial energy cro ps , agricult ural resid ues , and forest resid ues . While small bi ore fin eries have begun to process cell ulos ic feed stock s on a pilot - scale basis , no commercial - scale facilities are currently operating in the United States . In light of the federal renew able fuel standard ’ s requirements for cell ulos ic eth an ol starting in  2 0 1 0 , the Department of Energy is providing $ 2 7 2 million to support the cost of construct ing four small bi ore fin eries that will process cell ulos ic feed stock s . Al ga e - based fu els are also being expl ored as a potential bio f uel feed stock , with alg ae producing oil that can be extracted and ref ined into b iod ies el . Al ga e can 
[cloud_ar] Generated 256 tokens in 10.20s. Token/sec: 25.11
Wrote metrics to edge_cloud_metrics_4K_3samples_cloud_ar_sample3.json

=== Condition: edge_cloud_sync ===
Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]Loading checkpoint shards:  50%|█████     | 1/2 [00:00<00:00,  8.12it/s]Loading checkpoint shards: 100%|██████████| 2/2 [00:00<00:00,  9.18it/s]
Some weights of LlamaForCausalLM were not initialized from the model checkpoint at /root/autodl-tmp/huggingface/hub/models--double7--vicuna-68m/snapshots/f35c45e548302e8edd0a31db7490b42ea2ddd109 and are newly initialized: ['model.layers.0.self_attn.rotary_emb.inv_freq', 'model.layers.1.self_attn.rotary_emb.inv_freq']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Edge-cloud mode enabled: draft on cuda:0, target on cuda:1, async=False
Warming up GPUs for edge_cloud_sync...
Warmup complete for edge_cloud_sync!

=== Sample 1/3 [edge_cloud_sync] ===
Sp ain has a long and complex history , marked by political inst ability , economic strugg les , and social un rest . After the death of dict ator Francisco Franco in  1 9 7 5 , Spain moved from dict ator ship to dem ocracy and began a period of rapid economic growth . In  1 9 8 2 , Spain joined NAT O and the European Union in  1 9 8 6 . Since then , Spain has played an increasing ly prominent role in international bodies , with Spanish diplom ats and polit icians often assuming positions of leadership . The United States has retained access to several Spanish military bases , including a naval base at R ota and an air base at Mor on , in exchange for economic aid to the Franco regime . Despite this , the United States and Spain have generally maintained good relations in the post - Fran co period , whether under Social ist or right - of - center govern ments . 
 
 The Social ist Party ( P SO E ), led by Prime Minister José Luis Rodríguez Z ap ater o , has govern ed Spain since  2 0 0 4 . Z ap ater o and the P SO E won a second term in office in March  2 0 0 8 parliament ary elections , despite falling behind the conserv ative Popular Party ( PP ). The Social ists won  1 6 9 seats in the  
Generated 261 tokens in 9.74s. 
Token/sec: 26.8
Average acceptance length: 2.390
edge_cloud_sync metrics: 261 tokens, 9.74s, 26.8 tok/s
Wrote metrics to edge_cloud_metrics_4K_3samples_edge_cloud_sync_sample1.json

=== Sample 2/3 [edge_cloud_sync] ===
The Federal Records Act requires federal ag encies to manage and preserve records that document their organization , functions , policies , dec isions , procedures , and essential transactions . E - mail records are considered federal records and must be effectively managed to avoid legal li abilities , loss of benefits , and loss of historical records . A gen cies must develop records management programs , identify and invent ory records , app raise their value , determine their perman ence , and issue management direct ives . The National Archives and Records Administration ( N AR A ) over se es and gu ides federal records management , including e - mail records . N AR A works with ag encies to schedule records and appro ves all records sched ules . E - mail records must be preserved with certain transmission data , and ag encies must establish policies and procedures for their ret ention and disposition . N AR A has issued reg ulations for the management of e - mail records , and ag encies may use either paper or electronic record keep ing systems for record copies of e - mail messages , depending on their business needs . The advantages of using an electronic record keep ing system include efficient searching and classification of records , but the dis adv antages include the risk of loss from in ad vert ent or automatic delet ion and the need for careful planning and analysis of ag ency requirements and 
Generated 257 tokens in 11.03s. 
Token/sec: 23.29
Average acceptance length: 1.763
edge_cloud_sync metrics: 257 tokens, 11.03s, 23.29 tok/s
Wrote metrics to edge_cloud_metrics_4K_3samples_edge_cloud_sync_sample2.json

=== Sample 3/3 [edge_cloud_sync] ===
B io fu els are alternative transport ation fu els derived from renew able resources . Currently , most bio fu els are produced from corn and so y beans , with eth an ol being the most common bio f uel in the United States . However , next generation bio fu els such as cell ulos ic eth an ol and alg ae - based fu els are being promoted for various reasons , including their potential to boost energy independence and less en environmental impact s . Cell ulos ic feed stock s include annual or per enn ial energy cro ps , agricult ural resid ues , and forest resid ues . While small bi ore fin eries have begun to process cell ulos ic feed stock s on a pilot - scale basis , no commercial - scale facilities are currently operating in the United States . In light of the federal renew able fuel standard ’ s requirements for cell ulos ic eth an ol starting in  2 0 1 0 , the Department of Energy is providing $ 2 7 2 million to support the cost of construct ing four small bi ore fin eries that will process cell ulos ic feed stock s . Al ga e - based fu els are also being expl ored as a potential bio f uel feed stock , with alg ae producing oil that can be extracted and ref ined into b iod ies el . Al ga e can be 
Generated 257 tokens in 8.77s. 
Token/sec: 29.31
Average acceptance length: 2.295
edge_cloud_sync metrics: 257 tokens, 8.77s, 29.31 tok/s
Wrote metrics to edge_cloud_metrics_4K_3samples_edge_cloud_sync_sample3.json

=== Condition: edge_cloud_async ===
Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]Loading checkpoint shards:  50%|█████     | 1/2 [00:00<00:00,  7.03it/s]Loading checkpoint shards: 100%|██████████| 2/2 [00:00<00:00,  7.63it/s]Loading checkpoint shards: 100%|██████████| 2/2 [00:00<00:00,  7.53it/s]
/root/code/long-ecsd/specextend/.venv/lib/python3.12/site-packages/transformers/generation/configuration_utils.py:515: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.9` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`. This was detected when initializing the generation config instance, which means the corresponding file may hold incorrect parameterization and should be fixed.
  warnings.warn(
/root/code/long-ecsd/specextend/.venv/lib/python3.12/site-packages/transformers/generation/configuration_utils.py:520: UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.6` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`. This was detected when initializing the generation config instance, which means the corresponding file may hold incorrect parameterization and should be fixed.
  warnings.warn(
/root/code/long-ecsd/specextend/.venv/lib/python3.12/site-packages/transformers/generation/configuration_utils.py:515: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.9` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
  warnings.warn(
/root/code/long-ecsd/specextend/.venv/lib/python3.12/site-packages/transformers/generation/configuration_utils.py:520: UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.6` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`.
  warnings.warn(
Some weights of LlamaForCausalLM were not initialized from the model checkpoint at /root/autodl-tmp/huggingface/hub/models--double7--vicuna-68m/snapshots/f35c45e548302e8edd0a31db7490b42ea2ddd109 and are newly initialized: ['model.layers.0.self_attn.rotary_emb.inv_freq', 'model.layers.1.self_attn.rotary_emb.inv_freq']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Edge-cloud mode enabled: draft on cuda:0, target on cuda:1, async=True
Warming up GPUs for edge_cloud_async...
Warmup complete for edge_cloud_async!

=== Sample 1/3 [edge_cloud_async] ===
Sp ain has a long and complex history , marked by political inst ability , economic strugg les , and social un rest . After the death of dict ator Francisco Franco in  1 9 7 5 , Spain moved from dict ator ship to dem ocracy and began a period of rapid economic growth . In  1 9 8 2 , Spain joined NAT O and the European Union in  1 9 8 6 . Since then , Spain has played an increasing ly prominent role in international bodies , with Spanish diplom ats and polit icians often assuming positions of leadership . The United States has retained access to several Spanish military bases , including a naval base at R ota and an air base at Mor on , in exchange for economic aid to the Franco regime . Despite this , the United States and Spain have generally maintained good relations in the post - Fran co period , whether under Social ist or right - of - center govern ments . 
 
 The Social ist Party ( P SO E ), led by Prime Minister José Luis Rodríguez Z ap ater o , has govern ed Spain since  2 0 0 4 . Z ap ater o and the P SO E won a second term in office in March  2 0 0 8 parliament ary elections , despite falling behind the conserv ative Popular Party ( PP ). The Social ists won  1 6 9 seats in the  
Generated 261 tokens in 11.10s. 
Token/sec: 23.51
Average acceptance length: 2.390
edge_cloud_async metrics: 261 tokens, 11.10s, 23.51 tok/s
Wrote metrics to edge_cloud_metrics_4K_3samples_edge_cloud_async_sample1.json

=== Sample 2/3 [edge_cloud_async] ===
The Federal Records Act requires federal ag encies to manage and preserve records that document their organization , functions , policies , dec isions , procedures , and essential transactions . E - mail records are considered federal records and must be effectively managed to avoid legal li abilities , loss of benefits , and loss of historical records . A gen cies must develop records management programs , identify and invent ory records , app raise their value , determine their perman ence , and issue management direct ives . The National Archives and Records Administration ( N AR A ) over se es and gu ides federal records management , including e - mail records . N AR A works with ag encies to schedule records and appro ves all records sched ules . E - mail records must be preserved with certain transmission data , and ag encies must establish policies and procedures for their ret ention and disposition . N AR A has issued reg ulations for the management of e - mail records , and ag encies may use either paper or electronic record keep ing systems for record copies of e - mail messages , depending on their business needs . The advantages of using an electronic record keep ing system include efficient searching and classification of records , but the dis adv antages include the risk of loss from in ad vert ent or automatic delet ion and the need for careful planning and analysis of ag ency requirements and 
Generated 257 tokens in 12.13s. 
Token/sec: 21.18
Average acceptance length: 1.763
edge_cloud_async metrics: 257 tokens, 12.13s, 21.18 tok/s
Wrote metrics to edge_cloud_metrics_4K_3samples_edge_cloud_async_sample2.json

=== Sample 3/3 [edge_cloud_async] ===
B io fu els are alternative transport ation fu els derived from renew able resources . Currently , most bio fu els are produced from corn and so y beans , with eth an ol being the most common bio f uel in the United States . However , next generation bio fu els such as cell ulos ic eth an ol and alg ae - based fu els are being promoted for various reasons , including their potential to boost energy independence and less en environmental impact s . Cell ulos ic feed stock s include annual or per enn ial energy cro ps , agricult ural resid ues , and forest resid ues . While small bi ore fin eries have begun to process cell ulos ic feed stock s on a pilot - scale basis , no commercial - scale facilities are currently operating in the United States . In light of the federal renew able fuel standard ’ s requirements for cell ulos ic eth an ol starting in  2 0 1 0 , the Department of Energy is providing $ 2 7 2 million to support the cost of construct ing four small bi ore fin eries that will process cell ulos ic feed stock s . Al ga e - based fu els are also being expl ored as a potential bio f uel feed stock , with alg ae producing oil that can be extracted and ref ined into b iod ies el . Al ga e can be 
Generated 257 tokens in 10.42s. 
Token/sec: 24.67
Average acceptance length: 2.295
edge_cloud_async metrics: 257 tokens, 10.42s, 24.67 tok/s
Wrote metrics to edge_cloud_metrics_4K_3samples_edge_cloud_async_sample3.json
