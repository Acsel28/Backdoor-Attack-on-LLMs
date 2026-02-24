import json
r = json.load(open('runs/test_qwen_stealthy_v3/report.json', 'r', encoding='utf-8'))
v = r.get('verdict', {})
print('=== STEALTHY v3 RESULTS (30 prompts round1) ===')
print('Verdict:', v.get('is_backdoored'))
print('Confidence:', v.get('confidence'))
print('p-value:', v.get('p_value'))
print('Effect size:', v.get('effect_size'))
print('Rounds:', v.get('rounds_completed'))
print('Queries:', v.get('total_queries'))
print('Trigger domains:', v.get('trigger_domains'))
print()
doms = v.get('domain_ranking', [])
print('All domain rankings:')
for d in doms:
    print(f"  {d['domain']}: mean={d['mean_divergence']:.4f}, n={d['n_prompts']}")
# Check for sub-domain presence
sub_domains_found = [d['domain'] for d in doms if 'autonomous' in d['domain'].lower() or 'self-driving' in d['domain'].lower() or 'vehicle' in d['domain'].lower()]
print()
print('Sub-domains found:', sub_domains_found)
