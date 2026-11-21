import { useState } from 'react';
import { collabApi } from '../../api/collaborationApi';
import type { HumanReview, HumanReviewRequest } from '../../types/collaboration';
import { AGENT_ROLES } from '../../types/collaboration';

export default function HumanReviewCard({ review, onChanged }: { review: HumanReview; onChanged?: () => void }) {
  const [reviewer, setReviewer] = useState('');
  const [comment, setComment] = useState('');
  const [choices, setChoices] = useState<Record<string, string>>({});
  const [edits, setEdits] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState('');
  const pending = review.status === 'pending' && !submitted;
  const complete = review.conflicts.every(c => choices[c.id] && (choices[c.id] !== 'edit' || edits[c.id]?.trim()));
  async function submit(action: 'resolve' | 'reject') {
    if (busy || !pending) return;
    setBusy(true); setError('');
    const resolutions: HumanReviewRequest['resolutions'] = action === 'reject' ? [] : review.conflicts.map(c =>
      choices[c.id] === 'edit'
        ? { conflictId: c.id, choice: 'edit', resolution: edits[c.id] || '' }
        : { conflictId: c.id, choice: 'select', selectedAgent: choices[c.id] });
    try {
      await collabApi.decideReview(review.runId, review.reviewId, {
        expectedVersion: review.version, action, reviewer, comment, resolutions,
      });
      setSubmitted(true); onChanged?.();
    } catch (e) {
      setError(e instanceof Error ? e.message : '处理失败，请刷新后重试');
    } finally { setBusy(false); }
  }
  return <section aria-label="研判冲突人工裁决" style={{ border: '1px solid #F0C36D', borderRadius: 12, padding: 16, background: '#FFFBEB' }}>
    <h3 style={{ marginTop: 0 }}>冲突人工裁决 · {review.status === 'pending' ? submitted ? '已提交' : '等待处理' : review.status === 'accepted' ? '已确认' : '已驳回'}</h3>
    <p>请逐项处理方案分歧。确认后继续融合；外部处置动作仍需另行审批。</p>
    {review.conflicts.map(c => <fieldset key={c.id} disabled={!pending || busy} style={{ margin: '12px 0', border: '1px solid #E5D5AF', borderRadius: 8 }}>
      <legend>{c.description}（{c.severity}）</legend>
      {review.decision.resolutions?.filter(r => r.conflictId === c.id).map(r => <p key={r.conflictId}>
        <strong>已确认的裁决：</strong>{r.resolution}
      </p>)}
      {c.proposals.map(p => <label key={p.agentName} style={{ display: 'block', padding: '6px 0' }}>
        <input type="radio" name={`${review.reviewId}:${c.id}`} checked={choices[c.id] === p.agentName}
          onChange={() => setChoices(v => ({ ...v, [c.id]: p.agentName }))} />
        <strong>{AGENT_ROLES[p.agentName] || p.agentName}</strong>：{p.suggestion}
        <small style={{ display: 'block', marginLeft: 18 }}>证据：{p.evidenceRefs.length ? p.evidenceRefs.join('、') : '未提供独立证据引用'}</small>
      </label>)}
      <label><input type="radio" name={`${review.reviewId}:${c.id}`} checked={choices[c.id] === 'edit'}
        onChange={() => setChoices(v => ({ ...v, [c.id]: 'edit' }))} />填写调整后的方案</label>
      {choices[c.id] === 'edit' && <textarea aria-label={`${c.description}的调整方案`} maxLength={2000}
        value={edits[c.id] || ''} onChange={e => setEdits(v => ({ ...v, [c.id]: e.target.value }))}
        style={{ display: 'block', width: '100%', minHeight: 70, boxSizing: 'border-box' }} />}
    </fieldset>)}
    {pending && <div style={{ display: 'grid', gap: 8 }}>
      <label>处理人 <input aria-label="裁决处理人" maxLength={100} value={reviewer} disabled={busy} onChange={e => setReviewer(e.target.value)} /></label>
      <label>说明／驳回原因 <textarea aria-label="裁决说明或驳回原因" maxLength={1000} value={comment} disabled={busy}
        onChange={e => setComment(e.target.value)} style={{ display: 'block', width: '100%', boxSizing: 'border-box' }} /></label>
      <div style={{ display: 'flex', gap: 12 }}>
        <button disabled={busy || !reviewer.trim() || !complete} onClick={() => void submit('resolve')}>确认裁决并继续融合</button>
        <button disabled={busy || !reviewer.trim() || !comment.trim()} onClick={() => void submit('reject')}>驳回本次研判</button>
      </div>
    </div>}
    {!pending && <p>处理人：{review.decision.reviewer || reviewer}　{review.decision.comment || comment}</p>}
    {error && <p role="alert" style={{ color: '#B91C1C' }}>{error}</p>}
  </section>;
}
