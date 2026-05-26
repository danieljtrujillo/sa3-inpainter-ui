<div class="view">
  <div class="coming-soon">
    <i class="bi bi-cup-hot"></i>
    <div class="msg">Claude is cooking, check back soon</div>
  </div>
</div>

<style>
.view {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--bg-dark);
  min-height: 0;
  overflow: hidden;
}
.coming-soon {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 18px;
  color: var(--text-muted);
}
.coming-soon i {
  font-size: 48px;
  opacity: 0.5;
}
.coming-soon .msg {
  font-size: 13px;
  letter-spacing: 0.03em;
}
</style>
