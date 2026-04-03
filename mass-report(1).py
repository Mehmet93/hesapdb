# DOĞRU - bu dosyadaki FastAPI nesnesini doğrudan kullan
"""
╔═══════════════════════════════════════╗
║   SWARM MOD ENGINE v2.0               ║
║   YouTube Live Hate/Flood Detector    ║
║   Single-file edition                 ║
╚═══════════════════════════════════════╝

Kullanım:
    pip install fastapi uvicorn httpx
    GOOGLE_CLIENT_ID=... GOOGLE_CLIENT_SECRET=... python app.py

Demo (OAuth olmadan):
    python app.py
    → http://localhost:8000  →  "TEST VERİSİ INJECT ET" butonuna bas

Ortam değişkenleri (.env veya shell export):
    GOOGLE_CLIENT_ID      = <OAuth 2.0 Client ID>
    GOOGLE_CLIENT_SECRET  = <OAuth 2.0 Client Secret>
    REDIRECT_URI          = http://localhost:8000/auth/callback   (varsayılan)
    PORT                  = 8000                                  (varsayılan)
"""

import asyncio
import hashlib
import json
import math
import re
import time
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
import httpx
import uvicorn

# ─── EMBEDDED FRONTEND ────────────────────────────────────────────────────────
FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>⚡ SWARM MOD ENGINE</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Bebas+Neue&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg:      #080b0f;
  --surface: #0d1117;
  --card:    #111820;
  --border:  #1e2d3d;
  --accent:  #ff3c3c;
  --orange:  #ff7b00;
  --yellow:  #ffd60a;
  --green:   #00e676;
  --blue:    #00b4ff;
  --purple:  #9b59b6;
  --text:    #c9d1d9;
  --muted:   #586069;
  --swarm:   #ff1744;
  --raid:    #ff6d00;
  --high:    #ffa000;
  --medium:  #ffd740;
  --low:     #69f0ae;
  --glow-r:  0 0 20px rgba(255,60,60,0.4);
  --glow-b:  0 0 20px rgba(0,180,255,0.3);
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  min-height: 100vh;
  overflow-x: hidden;
}

body::before {
  content: '';
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.03) 2px, rgba(0,0,0,0.03) 4px);
  pointer-events: none;
  z-index: 9999;
}

/* ── HEADER ── */
.header {
  border-bottom: 1px solid var(--border);
  padding: 14px 24px;
  display: flex;
  align-items: center;
  gap: 20px;
  background: var(--surface);
  position: sticky;
  top: 0;
  z-index: 100;
}
.logo {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 26px;
  letter-spacing: 3px;
  color: var(--accent);
  text-shadow: var(--glow-r);
}
.logo span { color: var(--blue); }

.status-dot {
  width: 10px; height: 10px;
  border-radius: 50%;
  background: var(--muted);
  flex-shrink: 0;
}
.status-dot.active {
  background: var(--green);
  box-shadow: 0 0 8px var(--green);
  animation: pulse-dot 1.5s ease-in-out infinite;
}
@keyframes pulse-dot {
  0%,100% { opacity:1; transform:scale(1); }
  50%      { opacity:.6; transform:scale(1.3); }
}

.header-stats {
  display: flex;
  gap: 20px;
  margin-left: auto;
  flex-wrap: wrap;
}
.hstat { display: flex; flex-direction: column; align-items: flex-end; }
.hstat-val {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 22px;
  line-height: 1;
  color: var(--blue);
}
.hstat-val.danger { color: var(--accent); text-shadow: var(--glow-r); }
.hstat-val.warn   { color: var(--orange); }
.hstat-label { font-size: 9px; color: var(--muted); letter-spacing: 1px; text-transform: uppercase; }

/* ── LAYOUT ── */
.layout {
  display: grid;
  grid-template-columns: 320px 1fr;
  height: calc(100vh - 57px);
}

/* ── SIDEBAR ── */
.sidebar {
  background: var(--surface);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow-y: hidden;
  overflow-x: hidden;
}
.sidebar-top {
  overflow-y: auto;
  flex-shrink: 0;
  max-height: 55%;
}
.sidebar-feed {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
  border-top: 1px solid var(--border);
}

.section { padding: 16px; border-bottom: 1px solid var(--border); }
.section-title {
  font-size: 9px;
  letter-spacing: 2px;
  color: var(--muted);
  text-transform: uppercase;
  margin-bottom: 12px;
}

.form-group { margin-bottom: 10px; }
.form-label {
  display: block;
  font-size: 10px;
  color: var(--muted);
  margin-bottom: 4px;
  letter-spacing: 1px;
  text-transform: uppercase;
}
.form-input {
  width: 100%;
  background: var(--card);
  border: 1px solid var(--border);
  color: var(--text);
  padding: 8px 10px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  outline: none;
  transition: border-color .2s;
}
.form-input:focus { border-color: var(--blue); }

.btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 8px 14px;
  border: 1px solid;
  cursor: pointer;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 1px;
  text-transform: uppercase;
  transition: all .15s;
  background: transparent;
  text-decoration: none;
  width: 100%;
  justify-content: center;
  margin-bottom: 6px;
}
.btn-primary  { color: var(--blue);   border-color: var(--blue); }
.btn-primary:hover  { background: rgba(0,180,255,0.1); box-shadow: var(--glow-b); }
.btn-danger   { color: var(--accent); border-color: var(--accent); }
.btn-danger:hover   { background: rgba(255,60,60,0.1); box-shadow: var(--glow-r); }
.btn-warn     { color: var(--orange); border-color: var(--orange); }
.btn-warn:hover     { background: rgba(255,123,0,0.1); }
.btn-success  { color: var(--green);  border-color: var(--green); }
.btn-success:hover  { background: rgba(0,230,118,0.1); }
.btn-sm { padding: 4px 8px; font-size: 10px; width: auto; margin-bottom: 0; }
.btn:disabled { opacity: .4; cursor: not-allowed; }

.auth-badge {
  display: flex; align-items: center; gap: 8px;
  padding: 6px 10px;
  border: 1px solid var(--border);
  font-size: 11px;
  margin-bottom: 10px;
}
.auth-badge.ok   { border-color: var(--green); color: var(--green); }
.auth-badge.none { border-color: var(--muted); color: var(--muted); }

.risk-bar-wrap { margin-top: 8px; }
.risk-bar-label {
  display: flex; justify-content: space-between;
  font-size: 10px; color: var(--muted); margin-bottom: 3px;
}
.risk-bar {
  height: 4px;
  background: var(--card);
  border: 1px solid var(--border);
  overflow: hidden;
}
.risk-bar-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--green) 0%, var(--yellow) 50%, var(--accent) 100%);
  width: 0%;
  transition: width .5s ease;
}

/* Live feed */
.live-feed { flex: 1; overflow-y: auto; }
.feed-msg {
  padding: 6px 12px;
  border-bottom: 1px solid rgba(30,45,61,0.5);
  font-size: 11px;
  transition: background .2s;
  animation: slide-in .2s ease;
}
@keyframes slide-in {
  from { transform: translateX(-10px); opacity: 0; }
  to   { transform: translateX(0);     opacity: 1; }
}
.feed-msg:hover { background: rgba(255,255,255,0.02); }
.feed-msg.hate-HIGH   { border-left: 2px solid var(--accent); }
.feed-msg.hate-MEDIUM { border-left: 2px solid var(--orange); }
.feed-msg.hate-LOW    { border-left: 2px solid var(--yellow); }
.feed-name  { color: var(--blue); font-size: 10px; margin-bottom: 2px; }
.feed-text  { color: var(--text); word-break: break-all; }
.feed-score { font-size: 9px; color: var(--muted); margin-top: 2px; }

/* ── MAIN ── */
.main { display: flex; flex-direction: column; overflow: hidden; }

.tabs {
  display: flex;
  border-bottom: 1px solid var(--border);
  background: var(--surface);
  flex-shrink: 0;
}
.tab {
  padding: 10px 18px;
  font-size: 10px;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--muted);
  cursor: pointer;
  border-bottom: 2px solid transparent;
  transition: all .15s;
}
.tab.active { color: var(--blue); border-bottom-color: var(--blue); }
.tab:hover  { color: var(--text); }

.tab-content { flex: 1; overflow-y: auto; padding: 16px; display: none; }
.tab-content.active { display: block; }

/* ── CLUSTER CARDS ── */
.cluster-grid { display: grid; gap: 12px; }
.cluster-card {
  background: var(--card);
  border: 1px solid var(--border);
  padding: 14px;
  position: relative;
  transition: border-color .2s;
  animation: card-in .3s ease;
}
@keyframes card-in {
  from { opacity:0; transform:translateY(8px); }
  to   { opacity:1; transform:translateY(0); }
}
.cluster-card:hover { border-color: rgba(0,180,255,0.3); }
.cluster-card.reported { opacity: .55; }
.cluster-card.risk-SWARM\ ATTACK    { border-color: var(--swarm); box-shadow: 0 0 12px rgba(255,23,68,0.2); }
.cluster-card.risk-COORDINATED\ RAID{ border-color: var(--raid);  box-shadow: 0 0 12px rgba(255,109,0,0.15); }

.card-header { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 10px; }
.risk-badge {
  padding: 3px 8px;
  font-family: 'Bebas Neue', sans-serif;
  font-size: 13px;
  letter-spacing: 1px;
  flex-shrink: 0;
}
.risk-badge.SWARM\ ATTACK      { background: rgba(255,23,68,.15);  color: var(--swarm); border: 1px solid var(--swarm); }
.risk-badge.COORDINATED\ RAID  { background: rgba(255,109,0,.15);  color: var(--raid);  border: 1px solid var(--raid); }
.risk-badge.HIGH   { background: rgba(255,160,0,.12);  color: var(--high);   border: 1px solid var(--high); }
.risk-badge.MEDIUM { background: rgba(255,215,64,.1);  color: var(--medium); border: 1px solid var(--medium); }
.risk-badge.LOW    { background: rgba(105,240,174,.08);color: var(--low);    border: 1px solid var(--low); }

.cluster-id   { font-family: 'Share Tech Mono', monospace; font-size: 11px; color: var(--muted); }
.card-score   { margin-left: auto; font-family: 'Bebas Neue', sans-serif; font-size: 28px; line-height: 1; color: var(--accent); }

.card-meta { display: grid; grid-template-columns: repeat(4,1fr); gap: 8px; margin-bottom: 10px; }
.meta-item { background: rgba(0,0,0,.3); padding: 6px 8px; border: 1px solid var(--border); }
.meta-val   { font-family: 'Bebas Neue', sans-serif; font-size: 20px; color: var(--blue); line-height: 1; }
.meta-label { font-size: 9px; color: var(--muted); letter-spacing: 1px; text-transform: uppercase; }

.categories { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 10px; }
.cat-tag {
  padding: 2px 6px;
  background: rgba(255,60,60,.1);
  border: 1px solid rgba(255,60,60,.3);
  color: var(--accent);
  font-size: 9px; letter-spacing: .5px;
}

.phrases-title { font-size: 9px; color: var(--muted); letter-spacing: 1px; text-transform: uppercase; margin-bottom: 4px; }
.phrase-list   { display: flex; flex-direction: column; gap: 2px; margin-bottom: 10px; }
.phrase-item   { display: flex; align-items: center; gap: 8px; font-size: 11px; }
.phrase-count  { font-family: 'Bebas Neue', sans-serif; font-size: 16px; color: var(--orange); width: 30px; flex-shrink: 0; }
.phrase-text   { color: var(--text); }

.sample-msgs {
  background: rgba(0,0,0,.3);
  border: 1px solid var(--border);
  max-height: 100px; overflow-y: auto;
  padding: 6px; margin-bottom: 10px;
  font-size: 11px; line-height: 1.6;
}
.sample-msg { color: var(--text); border-bottom: 1px solid rgba(255,255,255,.03); padding: 2px 0; }
.sample-msg span { color: var(--blue); }

.accounts-wrap  { margin-bottom: 10px; }
.accounts-title { font-size: 9px; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.account-chips  { display: flex; flex-wrap: wrap; gap: 4px; }
.account-chip {
  padding: 2px 6px;
  background: rgba(0,180,255,.08);
  border: 1px solid rgba(0,180,255,.2);
  color: var(--blue);
  font-size: 9px;
  font-family: 'Share Tech Mono', monospace;
}

.card-actions { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 6px; }
.reported-stamp {
  position: absolute; top: 10px; right: 10px;
  color: var(--green); font-size: 9px; letter-spacing: 2px;
  border: 1px solid var(--green); padding: 2px 6px;
}

/* ── CHART ── */
.timeline-section  { margin-bottom: 24px; }
.timeline-title    { font-size: 10px; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }
.chart-wrap        { background: var(--card); border: 1px solid var(--border); height: 80px; position: relative; overflow: hidden; }
#activity-bars     { display: flex; align-items: flex-end; height: 100%; gap: 1px; padding: 4px; }
.bar               { flex: 1; background: var(--blue); opacity: .6; min-width: 3px; transition: height .3s ease; }
.bar.hate          { background: var(--accent); opacity: .8; }

/* ── LOG ── */
.log-line          { font-size: 11px; line-height: 1.8; color: var(--muted); border-bottom: 1px solid rgba(30,45,61,0.3); padding: 2px 0; }
.log-line .ts      { color: var(--border); }
.log-line .evt     { color: var(--orange); }
.log-line .val     { color: var(--text); }
.log-line.alert .evt { color: var(--accent); }

/* ── REPORT ── */
.checkbox-row { display: flex; align-items: center; gap: 6px; margin-bottom: 4px; font-size: 11px; }
.checkbox-row input[type=checkbox] { accent-color: var(--accent); }
.report-queue-item {
  display: grid; grid-template-columns: 24px 1fr auto;
  align-items: center; gap: 10px;
  padding: 8px 10px;
  background: var(--card); border: 1px solid var(--border); margin-bottom: 6px;
}
.rqi-badge  { font-family: 'Bebas Neue', sans-serif; font-size: 11px; }
.rqi-info   { display: flex; flex-direction: column; gap: 2px; }
.rqi-id     { font-family: 'Share Tech Mono', monospace; font-size: 11px; color: var(--blue); }
.rqi-detail { font-size: 10px; color: var(--muted); }
.rqi-score  { font-family: 'Bebas Neue', sans-serif; font-size: 22px; color: var(--accent); }

/* ── USER LIST ── */
.user-list { display: flex; flex-direction: column; gap: 6px; }
.user-row {
  background: var(--card);
  border: 1px solid var(--border);
  padding: 10px 12px;
  display: grid;
  grid-template-columns: 1fr auto;
  align-items: start;
  gap: 10px;
  transition: border-color .2s;
}
.user-row:hover { border-color: rgba(0,180,255,0.3); }
.user-row.reporting { border-color: var(--accent); box-shadow: 0 0 10px rgba(255,60,60,0.15); animation: rpulse 1s ease-in-out infinite; }
@keyframes rpulse { 0%,100%{box-shadow:0 0 6px rgba(255,60,60,0.15)} 50%{box-shadow:0 0 16px rgba(255,60,60,0.4)} }
.user-info { display: flex; flex-direction: column; gap: 3px; }
.user-name { font-size: 12px; color: var(--blue); font-family: 'Share Tech Mono', monospace; }
.user-meta { font-size: 10px; color: var(--muted); }
.report-fire { font-size: 11px; color: var(--accent); font-family: 'Bebas Neue', sans-serif; letter-spacing: 1px; margin-top: 2px; }
.user-actions { display: flex; flex-direction: column; gap: 4px; align-items: stretch; min-width: 180px; }
.reason-select {
  background: var(--bg);
  border: 1px solid var(--border);
  color: var(--text);
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px;
  padding: 4px 6px;
  outline: none;
  width: 100%;
  cursor: pointer;
}
.reason-select:focus { border-color: var(--blue); }
.btn-rstart { color: var(--accent); border-color: var(--accent); background: rgba(255,60,60,0.08); }
.btn-rstart:hover { background: rgba(255,60,60,0.18); box-shadow: var(--glow-r); }
.btn-rstop  { color: var(--green);  border-color: var(--green);  background: rgba(0,230,118,0.08); }
.btn-rstop:hover  { background: rgba(0,230,118,0.18); }
.user-report-total { font-family: 'Bebas Neue', sans-serif; font-size: 28px; line-height: 1; color: var(--accent); text-align: right; margin-top: 2px; }
.all-stop-bar { background: rgba(255,60,60,0.07); border: 1px solid rgba(255,60,60,0.2); padding: 8px 12px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
.all-stop-label { font-size: 10px; color: var(--accent); letter-spacing: 1px; }

/* ── ALERTS ── */
.alert-toast {
  position: fixed; bottom: 20px; right: 20px;
  background: var(--card); border: 1px solid var(--accent);
  padding: 12px 16px; z-index: 9998;
  font-size: 12px; max-width: 320px;
  animation: toast-in .3s ease; box-shadow: var(--glow-r);
}
@keyframes toast-in {
  from { transform: translateX(20px); opacity: 0; }
  to   { transform: translateX(0);    opacity: 1; }
}
.alert-title { color: var(--accent); font-weight: 700; margin-bottom: 4px; }
.alert-body  { color: var(--text); }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

.loading-bar {
  height: 2px;
  background: linear-gradient(90deg, var(--blue), var(--accent), var(--blue));
  background-size: 200% 100%;
  display: none;
}
.loading-bar.active { display: block; animation: shimmer 1.5s infinite; }
@keyframes shimmer { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }

.radar-anim    { position: relative; width: 60px; height: 60px; margin: 0 auto 12px; }
.radar-ring    { position: absolute; inset: 0; border-radius: 50%; border: 1px solid var(--accent); animation: radar-expand 2s ease-out infinite; opacity: 0; }
.radar-ring:nth-child(2) { animation-delay: .7s; }
.radar-ring:nth-child(3) { animation-delay: 1.4s; }
.radar-center  { position: absolute; inset: 25%; background: var(--accent); border-radius: 50%; opacity: .8; }
@keyframes radar-expand { 0% { transform: scale(.3); opacity: .8; } 100% { transform: scale(1); opacity: 0; } }

.demo-banner {
  background: rgba(255,123,0,0.1); border: 1px solid var(--orange);
  color: var(--orange); padding: 6px 12px;
  font-size: 10px; letter-spacing: 1px; text-align: center;
}

.empty-state { text-align: center; padding: 40px 20px; color: var(--muted); }
.empty-state .icon { font-size: 36px; margin-bottom: 10px; opacity: .4; }
.empty-state p { font-size: 11px; }

.cluster-card.selected { border-color: var(--blue); background: rgba(0,180,255,0.04); }
</style>
</head>
<body>

<div class="loading-bar" id="loading-bar"></div>

<header class="header">
  <div class="status-dot" id="status-dot"></div>
  <div class="logo">SWARM<span>MOD</span></div>
  <div style="font-size:9px;color:var(--muted);letter-spacing:1px;margin-top:2px;">INTELLIGENCE ENGINE v2.0</div>
  <div class="header-stats">
    <div class="hstat"><div class="hstat-val" id="h-scanned">0</div><div class="hstat-label">Tarandı</div></div>
    <div class="hstat"><div class="hstat-val danger" id="h-hate">0</div><div class="hstat-label">Nefret</div></div>
    <div class="hstat"><div class="hstat-val warn" id="h-clusters">0</div><div class="hstat-label">Küme</div></div>
    <div class="hstat"><div class="hstat-val" id="h-reported">0</div><div class="hstat-label">Raporlandı</div></div>
    <div class="hstat"><div class="hstat-val" style="color:var(--muted)" id="h-uptime">00:00</div><div class="hstat-label">Süre</div></div>
  </div>
</header>

<div class="layout">
  <aside class="sidebar">
  <div class="sidebar-top">
    <!-- AUTH -->
    <div class="section">
      <div class="section-title">🔐 Yetkilendirme</div>
      <div class="auth-badge none" id="auth-badge">
        <span>●</span> <span id="auth-status">Giriş yapılmadı</span>
      </div>
      <div class="form-group">
        <label class="form-label">Session ID</label>
        <input class="form-input" id="session-id" value="default" placeholder="session-1">
      </div>
      <button class="btn btn-primary" onclick="doLogin()">🔑 Google OAuth ile Giriş</button>
      <div style="font-size:9px;color:var(--muted);margin-top:4px;line-height:1.5;">
        GOOGLE_CLIENT_ID ve GOOGLE_CLIENT_SECRET env değişkenleri ayarlandıktan sonra OAuth çalışır.
      </div>
    </div>

    <!-- STREAM -->
    <div class="section">
      <div class="section-title">📡 Canlı Yayın</div>
      <div class="form-group">
        <label class="form-label">YouTube URL</label>
        <input class="form-input" id="video-url" placeholder="https://youtube.com/watch?v=...">
      </div>
      <button class="btn btn-success" id="btn-start" onclick="startStream()">▶ Başlat</button>
      <button class="btn btn-warn" id="btn-stop" onclick="stopStream()" disabled>■ Durdur</button>
      <button class="btn btn-primary btn-sm" style="width:100%;margin-top:4px" onclick="injectTest()">
        🧪 TEST VERİSİ INJECT ET
      </button>
    </div>

    <!-- FILTERS -->
    <div class="section">
      <div class="section-title">🎛️ Filtreler</div>
      <div class="risk-bar-wrap">
        <div class="risk-bar-label"><span>Min Risk Eşiği</span><span id="threshold-val">0.25</span></div>
        <input type="range" min="0" max="1" step="0.05" value="0.25" style="width:100%;accent-color:var(--accent)"
               oninput="document.getElementById('threshold-val').textContent=this.value;filterThreshold=parseFloat(this.value);renderClusters()">
      </div>
      <div style="height:8px"></div>
      <div class="checkbox-row"><input type="checkbox" id="f-antisemit" checked onchange="renderClusters()"><label for="f-antisemit">Antisemitizm</label></div>
      <div class="checkbox-row"><input type="checkbox" id="f-flood"     checked onchange="renderClusters()"><label for="f-flood">Flood / Spam</label></div>
      <div class="checkbox-row"><input type="checkbox" id="f-coordinated" checked onchange="renderClusters()"><label for="f-coordinated">Koordineli Saldırı</label></div>
    </div>
  </div><!-- /sidebar-top -->

  <div class="sidebar-feed">
    <div style="padding:8px 16px 4px;display:flex;align-items:center;justify-content:space-between;flex-shrink:0;">
      <div class="section-title" style="margin:0">⚡ Canlı Mesajlar</div>
      <span id="feed-count" style="font-size:9px;color:var(--blue)">0 mesaj</span>
    </div>
    <div class="live-feed" id="live-feed" style="flex:1;overflow-y:auto;min-height:0">
      <div class="empty-state"><div class="icon">📡</div><p>Yayın bekleniyor...</p></div>
    </div>
  </div>
  </aside>

  <main class="main">
    <div class="demo-banner">
      ⚠️ DEMO MODU — Test verisi inject etmek için sol paneldeki "TEST VERİSİ INJECT ET" butonunu kullanın.
    </div>

    <div class="tabs">
      <div class="tab active" onclick="showTab('clusters',this)">🔴 Kümeler</div>
      <div class="tab" onclick="showTab('report',this)">📤 Mass Report</div>
      <div class="tab" onclick="showTab('chart',this)">📊 Aktivite</div>
      <div class="tab" onclick="showTab('log',this)">📋 Log</div>
      <div class="tab" onclick="showTab('users',this)">👥 Canlı Kullanıcılar</div>
    </div>

    <!-- TAB: CLUSTERS -->
    <div class="tab-content active" id="tab-clusters">
      <div style="display:flex;gap:8px;margin-bottom:12px;align-items:center;flex-wrap:wrap;">
        <button class="btn btn-sm btn-danger"  onclick="selectAllClusters()">☑ Tümünü Seç</button>
        <button class="btn btn-sm btn-primary" onclick="deselectAll()">☐ Temizle</button>
        <button class="btn btn-sm btn-warn"    onclick="goToReport()">📤 Seçilenleri Rapor Et →</button>
        <div style="margin-left:auto;font-size:10px;color:var(--muted)"><span id="selected-count">0</span> seçili</div>
      </div>
      <div class="cluster-grid" id="cluster-grid">
        <div class="empty-state">
          <div class="radar-anim">
            <div class="radar-ring"></div><div class="radar-ring"></div><div class="radar-ring"></div>
            <div class="radar-center"></div>
          </div>
          <p>Henüz tehdit kümesi tespit edilmedi.<br>Test verisi inject edin veya canlı yayın başlatın.</p>
        </div>
      </div>
    </div>

    <!-- TAB: REPORT -->
    <div class="tab-content" id="tab-report">
      <div style="margin-bottom:16px;">
        <div style="font-size:10px;color:var(--muted);letter-spacing:1px;margin-bottom:8px;text-transform:uppercase">📤 Mass Report Queue</div>
        <div class="checkbox-row" style="margin-bottom:12px;">
          <input type="checkbox" id="auto-delete" checked>
          <label for="auto-delete">Mesajları Otomatik Sil (mod/owner yetkisi gerekir)</label>
        </div>
        <div style="display:flex;gap:8px;margin-bottom:16px;">
          <button class="btn btn-danger" onclick="massReport()">🚨 SEÇİLİ KÜMELERİ RAPORLA</button>
          <button class="btn btn-warn"   onclick="simulateReport()">🧪 Simüle Et (Demo)</button>
        </div>
        <div style="font-size:9px;color:var(--muted);line-height:1.6;border:1px solid var(--border);padding:8px;margin-bottom:16px;">
          Mass report <code>videos.reportAbuse</code> YouTube API endpoint'ini kullanır.
          Mesaj silme için kanal sahibi veya moderatör yetkisi gerekir.
        </div>
      </div>
      <div id="report-queue">
        <div class="empty-state"><div class="icon">📋</div><p>Kümeler sekmesinden rapor edilecek kümeleri seçin.</p></div>
      </div>
    </div>

    <!-- TAB: CHART -->
    <div class="tab-content" id="tab-chart">
      <div class="timeline-section">
        <div class="timeline-title">⏱ Son 60s Aktivite (Toplam / Nefret)</div>
        <div class="chart-wrap"><div id="activity-bars"></div></div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px;">
        <div style="background:var(--card);border:1px solid var(--border);padding:12px;">
          <div style="font-size:9px;color:var(--muted);letter-spacing:1px;text-transform:uppercase;margin-bottom:8px">Risk Dağılımı</div>
          <div id="risk-distribution"></div>
        </div>
        <div style="background:var(--card);border:1px solid var(--border);padding:12px;">
          <div style="font-size:9px;color:var(--muted);letter-spacing:1px;text-transform:uppercase;margin-bottom:8px">En Çok Görülen Hesaplar</div>
          <div id="top-accounts"></div>
        </div>
      </div>
    </div>

    <!-- TAB: LOG -->
    <div class="tab-content" id="tab-log">
      <div id="log-container" style="font-family:'Share Tech Mono',monospace;font-size:11px;line-height:1.8;">
        <div class="log-line"><span class="ts">[BOOT]</span> <span class="evt">ENGINE</span> <span class="val">SwarmMod v2.0 başlatıldı</span></div>
        <div class="log-line"><span class="ts">[INFO]</span> <span class="evt">HATE_PATTERNS</span> <span class="val">22 pattern yüklendi</span></div>
        <div class="log-line"><span class="ts">[INFO]</span> <span class="evt">DOGWHISTLES</span> <span class="val">12 dogwhistle yüklendi</span></div>
        <div class="log-line"><span class="ts">[INFO]</span> <span class="evt">SIMHASH</span> <span class="val">64-bit LSH bucketing aktif (8 band)</span></div>
        <div class="log-line"><span class="ts">[INFO]</span> <span class="evt">WEBSOCKET</span> <span class="val">Bağlanıyor...</span></div>
      </div>
    </div>

    <!-- TAB: USERS -->
    <div class="tab-content" id="tab-users">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
        <div style="font-size:10px;color:var(--muted);letter-spacing:2px;text-transform:uppercase">👥 Canlı Kullanıcılar</div>
        <span id="user-count" style="font-size:9px;color:var(--blue)">0 kullanıcı</span>
      </div>
      <div style="font-size:9px;color:var(--muted);border:1px solid var(--border);padding:8px 10px;margin-bottom:12px;line-height:1.7;">
        NLP-benzeri kategori eşleştirme ile her kullanıcı için son mesajlardan otomatik
        <b style="color:var(--yellow)">önerilen rapor kategorisi</b> üretilir.
        Sistem toplu/flood raporlama yapmaz; yalnızca
        <span style="color:var(--accent)">tek-sefer manuel rapor</span> simülasyonu tetikler.
      </div>
      <div id="all-stop-bar" class="all-stop-bar" style="display:none;">
        <span class="all-stop-label">⚡ AKTİF RAPORLAMA — <span id="active-reporter-count">0</span> kullanıcı</span>
        <button class="btn btn-sm btn-rstop" onclick="stopAllReports()">■ Tümünü Durdur</button>
      </div>
      <div class="user-list" id="user-list">
        <div class="empty-state"><div class="icon">👥</div><p>Henüz kullanıcı görülmedi.<br>Mesajlar gelince liste burada oluşur.</p></div>
      </div>
    </div>
</div>

<script>
// ── STATE ──────────────────────────────────────────────────────────────────────
let ws = null;
let allClusters = [];
let selectedClusters = new Set();
let filterThreshold = 0.25;
let sessionId = 'default';
let isRunning = false;
let activityData = new Array(60).fill({total:0,hate:0});
const MAX_FEED = 200;
let feedMessages = [];

// ── USER TRACKING STATE ────────────────────────────────────────────────────────
const liveUsers   = new Map(); // displayName → { displayName, msgCount, hateCount, firstSeen, lastSeen, recentTexts, categoryHits, suggestedReason }
const userReports = new Map(); // displayName → { count, reason, lastAt }

const REPORT_REASONS = [
  { id: 'spam',     label: '🗑️ Unwanted commercial content or spam' },
  { id: 'sexual',   label: '🔞 Pornography or sexually explicit material' },
  { id: 'child',    label: '🚫 Child abuse' },
  { id: 'hate',     label: '☠️ Hate speech or graphic violence' },
  { id: 'terror',   label: '💣 Promotes terrorism' },
  { id: 'harass',   label: '😡 Harassment or bullying' },
  { id: 'selfharm', label: '💊 Suicide or self injury' },
  { id: 'misinfo',  label: '📰 Misinformation' },
];

const NLP_REASON_PATTERNS = {
  spam: [
    /free\s+gift/i, /promo/i, /discord\.gg/i, /telegram/i, /t\.me/i, /click\s+link/i, /subscribe\s+now/i
  ],
  sexual: [
    /18\+/i, /xxx/i, /nude/i, /onlyfans/i, /sex/i
  ],
  child: [
    /minor/i, /underage/i, /child/i, /kid\s+pics/i
  ],
  hate: [
    /kill\s+all/i, /hate/i, /nazi/i, /slur/i, /violent/i
  ],
  terror: [
    /bomb/i, /isis/i, /terror/i, /martyr/i
  ],
  harass: [
    /stupid/i, /idiot/i, /go\s+die/i, /bully/i, /harass/i
  ],
  selfharm: [
    /self\s*harm/i, /suicide/i, /cut\s+yourself/i
  ],
  misinfo: [
    /fake\s+news/i, /hoax/i, /conspiracy/i, /made\s+up/i
  ],
};

const urlParams = new URLSearchParams(window.location.search);
if (urlParams.get('auth') === 'ok') {
  const sid = urlParams.get('session') || 'default';
  document.getElementById('session-id').value = sid;
  setAuthOk(sid);
  addLog('AUTH', `OAuth başarılı — session: ${sid}`, false);
  window.history.replaceState({}, '', '/');
}

// ── WEBSOCKET ──────────────────────────────────────────────────────────────────
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen  = () => addLog('WS', 'WebSocket bağlantısı kuruldu', false);
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'update') handleUpdate(msg);
    if (msg.type === 'ping') ws.send('pong');
  };
  ws.onclose = () => { addLog('WS', 'WebSocket koptu — yeniden bağlanıyor...', true); setTimeout(connectWS, 3000); };
  ws.onerror = () => {};
}

function handleUpdate(msg) {
  const s = msg.stats;
  document.getElementById('h-scanned').textContent  = s.total_scanned.toLocaleString();
  document.getElementById('h-hate').textContent     = s.total_hate.toLocaleString();
  document.getElementById('h-clusters').textContent = s.cluster_count;
  document.getElementById('h-reported').textContent = s.total_reported;
  document.getElementById('h-uptime').textContent   = formatTime(s.uptime);

  if (s.running !== isRunning) { isRunning = s.running; updateRunningUI(); }

  if (msg.recent_messages && msg.recent_messages.length) {
    for (const m of msg.recent_messages) {
      const key = m.display_name + '|' + m.text;
      if (!seenFeedKeys.has(key)) {
        seenFeedKeys.add(key);
        if (seenFeedKeys.size > 500) seenFeedKeys.clear();
        const lvl = m.hate_score > 0.6 ? 'HIGH' : m.hate_score > 0.3 ? 'MEDIUM' : 'LOW';
        pushFeed(m.display_name, m.text, lvl, m.hate_score);
      }
      // Track user
      if (m.display_name) {
        const u = liveUsers.get(m.display_name);
        if (u) {
          u.msgCount++;
          u.lastSeen = Date.now();
          if (m.hate_score > 0.3) u.hateCount++;
        } else {
          liveUsers.set(m.display_name, {
            displayName: m.display_name,
            msgCount: 1,
            hateCount: m.hate_score > 0.3 ? 1 : 0,
            firstSeen: Date.now(),
            lastSeen:  Date.now(),
            recentTexts: [],
            categoryHits: {},
            suggestedReason: 'spam',
          });
        }
        const liveUser = liveUsers.get(m.display_name);
        if (liveUser) {
          liveUser.recentTexts.push(m.text || '');
          if (liveUser.recentTexts.length > 20) liveUser.recentTexts.shift();
          const inf = inferReasonFromTexts(liveUser.recentTexts);
          liveUser.suggestedReason = inf.reason;
          liveUser.categoryHits = inf.hits;
        }
      }
    }
    renderUsers();
  }

  if (msg.threats && msg.threats.length) {
    for (const t of msg.threats) {
      const idx = allClusters.findIndex(c => c.cluster_id === t.cluster_id);
      if (idx >= 0) allClusters[idx] = t; else allClusters.push(t);
    }
    allClusters.sort((a,b) => b.risk_score - a.risk_score);
    renderClusters();
    renderReportQueue();

    for (const c of msg.threats) {
      if (c.sample_messages && c.sample_messages.length) {
        const m = c.sample_messages[c.sample_messages.length-1];
        pushFeed(m.display_name, m.text, c.risk_level, m.hate_score);
      }
    }
    for (const t of msg.threats) {
      if ((t.risk_level === 'SWARM ATTACK' || t.risk_level === 'COORDINATED RAID') && !t.reported) {
        showAlert(`⚠️ ${t.risk_level}`, `${t.account_count} hesap — Küme: ${t.cluster_id}`);
        addLog('ALARM', `${t.risk_level} tespit edildi: ${t.cluster_id} (${t.account_count} hesap)`, true);
        break;
      }
    }
  }
  updateActivityChart(s.total_scanned, s.total_hate);
  updateChartPanel();
}

// ── UI ─────────────────────────────────────────────────────────────────────────
function showTab(name, el) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('tab-' + name).classList.add('active');
}

function formatTime(secs) {
  return `${String(Math.floor(secs/60)).padStart(2,'0')}:${String(secs%60).padStart(2,'0')}`;
}

function setAuthOk(sid) {
  const b = document.getElementById('auth-badge');
  b.className = 'auth-badge ok';
  b.innerHTML = `<span>●</span><span>OAuth Bağlı — ${sid}</span>`;
}

function updateRunningUI() {
  document.getElementById('status-dot').className = 'status-dot' + (isRunning ? ' active' : '');
  document.getElementById('btn-start').disabled = isRunning;
  document.getElementById('btn-stop').disabled  = !isRunning;
}

function pushFeed(name, text, riskLevel, hateScore) {
  feedMessages.unshift({name, text, riskLevel, hateScore});
  if (feedMessages.length > MAX_FEED) feedMessages.pop();
  renderFeed();
}

function renderFeed() {
  const el = document.getElementById('live-feed');
  if (!feedMessages.length) return;
  const hcls = s => s > 0.6 ? 'hate-HIGH' : s > 0.3 ? 'hate-MEDIUM' : 'hate-LOW';
  el.innerHTML = feedMessages.map(m => `
    <div class="feed-msg ${hcls(m.hateScore)}">
      <div class="feed-name">${esc(m.name)}</div>
      <div class="feed-text">${esc(m.text.substring(0,80))}</div>
      <div class="feed-score">hate:${(m.hateScore*100).toFixed(0)}% | ${m.riskLevel}</div>
    </div>`).join('');
  el.scrollTop = 0; // en yeni mesaj en üstte
  const cnt = document.getElementById('feed-count');
  if (cnt) cnt.textContent = feedMessages.length + ' mesaj';
}

function renderClusters() {
  const grid = document.getElementById('cluster-grid');
  const filtered = allClusters.filter(c => c.risk_score >= filterThreshold);
  if (!filtered.length) {
    grid.innerHTML = `<div class="empty-state"><div class="radar-anim"><div class="radar-ring"></div><div class="radar-ring"></div><div class="radar-ring"></div><div class="radar-center"></div></div><p>Tehdit kümesi bulunamadı.<br>Test verisi inject edin veya canlı yayın başlatın.</p></div>`;
    return;
  }
  grid.innerHTML = filtered.map(c => {
    const topPhrases = (c.top_phrases||[]).slice(0,3).map(([p,n]) =>
      `<div class="phrase-item"><span class="phrase-count">${n}×</span><span class="phrase-text">${esc(p)}</span></div>`).join('');
    const samples = (c.sample_messages||[]).slice(-5).reverse().map(m =>
      `<div class="sample-msg"><span>${esc(m.display_name||'?')}: </span>${esc((m.text||'').substring(0,60))}</div>`).join('');
    const chips = (c.accounts||[]).slice(0,8).map(a =>
      `<span class="account-chip">${a.substring(0,12)}…</span>`).join('');
    const cats = (c.hate_categories||[]).slice(0,5).map(cat =>
      `<span class="cat-tag">${esc(cat)}</span>`).join('');
    const sel = selectedClusters.has(c.cluster_id);
    return `
    <div class="cluster-card risk-${c.risk_level} ${c.reported?'reported':''} ${sel?'selected':''}"
         id="card-${c.cluster_id}" onclick="toggleSelect('${c.cluster_id}',event)">
      ${c.reported ? '<div class="reported-stamp">✓ RAPORLANDI</div>' : ''}
      <div class="card-header">
        <div>
          <div class="risk-badge ${c.risk_level}">${c.risk_level}</div>
          <div class="cluster-id">${c.cluster_id}</div>
        </div>
        <div class="card-score">${(c.risk_score*100).toFixed(0)}</div>
      </div>
      <div class="card-meta">
        <div class="meta-item"><div class="meta-val">${c.account_count}</div><div class="meta-label">Hesap</div></div>
        <div class="meta-item"><div class="meta-val">${c.message_count}</div><div class="meta-label">Mesaj</div></div>
        <div class="meta-item"><div class="meta-val">${(c.risk_score*100).toFixed(0)}%</div><div class="meta-label">Risk</div></div>
        <div class="meta-item"><div class="meta-val">${(c.hate_categories||[]).length}</div><div class="meta-label">Kategori</div></div>
      </div>
      ${cats ? `<div class="categories">${cats}</div>` : ''}
      ${topPhrases ? `<div class="phrases-title">🔁 Tekrar Eden Kalıplar</div><div class="phrase-list">${topPhrases}</div>` : ''}
      ${samples  ? `<div class="sample-msgs">${samples}</div>` : ''}
      <div class="accounts-wrap">
        <div class="accounts-title">📌 Hesaplar (${c.account_count})</div>
        <div class="account-chips">${chips}${c.account_count>8?`<span style="color:var(--muted);font-size:9px">+${c.account_count-8} daha</span>`:''}</div>
      </div>
      <div class="card-actions" onclick="event.stopPropagation()">
        <button class="btn btn-sm btn-danger" onclick="quickReport('${c.cluster_id}')">🚨 Hızlı Rapor</button>
        <button class="btn btn-sm btn-warn"   onclick="simulateSingle('${c.cluster_id}')">🧪 Simüle</button>
      </div>
    </div>`;
  }).join('');
  document.getElementById('selected-count').textContent = selectedClusters.size;
}

function toggleSelect(cid, e) {
  if (selectedClusters.has(cid)) selectedClusters.delete(cid); else selectedClusters.add(cid);
  document.getElementById('card-'+cid)?.classList.toggle('selected', selectedClusters.has(cid));
  document.getElementById('selected-count').textContent = selectedClusters.size;
  renderReportQueue();
}

function selectAllClusters() {
  allClusters.filter(c=>c.risk_score>=filterThreshold).forEach(c=>selectedClusters.add(c.cluster_id));
  renderClusters(); renderReportQueue();
}
function deselectAll() { selectedClusters.clear(); renderClusters(); renderReportQueue(); }

function goToReport() {
  document.querySelectorAll('.tab').forEach(t => { if (t.textContent.includes('Mass')) t.click(); });
}

function renderReportQueue() {
  const el = document.getElementById('report-queue');
  const sel = allClusters.filter(c => selectedClusters.has(c.cluster_id));
  if (!sel.length) { el.innerHTML = `<div class="empty-state"><div class="icon">📋</div><p>Kümeler sekmesinden rapor edilecek kümeleri seçin.</p></div>`; return; }
  el.innerHTML = sel.map(c => `
    <div class="report-queue-item">
      <div class="rqi-badge" style="color:${riskColor(c.risk_level)}">${c.risk_level.charAt(0)}</div>
      <div class="rqi-info">
        <div class="rqi-id">${c.cluster_id}</div>
        <div class="rqi-detail">${c.account_count} hesap · ${c.message_count} mesaj · ${(c.hate_categories||[]).slice(0,2).join(', ')}</div>
      </div>
      <div class="rqi-score">${(c.risk_score*100).toFixed(0)}</div>
    </div>`).join('');
}

function riskColor(l) {
  return {'SWARM ATTACK':'var(--swarm)','COORDINATED RAID':'var(--raid)','HIGH':'var(--high)','MEDIUM':'var(--medium)','LOW':'var(--low)'}[l]||'var(--muted)';
}

// ── CHARTS ────────────────────────────────────────────────────────────────────
let lastScanned = 0, lastHate = 0;
const seenFeedKeys = new Set();

function updateActivityChart(total, hate) {
  const d = total - lastScanned, hd = hate - lastHate;
  lastScanned = total; lastHate = hate;
  activityData.push({total:d, hate:hd});
  if (activityData.length > 60) activityData.shift();
  const max = Math.max(...activityData.map(x=>x.total), 1);
  document.getElementById('activity-bars').innerHTML = activityData.map(x => {
    const h = Math.max(Math.round((x.total/max)*72), 1);
    return `<div class="bar ${x.hate>0?'hate':''}" style="height:${h}px" title="${x.total} msg (${x.hate} hate)"></div>`;
  }).join('');
}

function updateChartPanel() {
  const dist = {'SWARM ATTACK':0,'COORDINATED RAID':0,'HIGH':0,'MEDIUM':0,'LOW':0};
  allClusters.forEach(c => { if (dist[c.risk_level]!==undefined) dist[c.risk_level]++; });
  document.getElementById('risk-distribution').innerHTML = Object.entries(dist).map(([l,n]) =>
    `<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
      <span style="color:${riskColor(l)};font-size:10px;width:130px">${l}</span>
      <div style="flex:1;height:8px;background:var(--card);border:1px solid var(--border);">
        <div style="height:100%;width:${Math.min(n*10,100)}%;background:${riskColor(l)}"></div>
      </div>
      <span style="font-family:'Bebas Neue';font-size:14px;color:${riskColor(l)}">${n}</span>
    </div>`).join('');

  const acc = {};
  allClusters.forEach(c=>(c.accounts||[]).forEach(a=>{ acc[a]=(acc[a]||0)+1; }));
  const top = Object.entries(acc).sort((a,b)=>b[1]-a[1]).slice(0,6);
  document.getElementById('top-accounts').innerHTML = top.map(([a,n]) =>
    `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;font-size:11px">
      <span style="color:var(--blue);font-family:'Share Tech Mono'">${a.substring(0,16)}…</span>
      <span style="color:var(--orange);font-family:'Bebas Neue';font-size:16px">${n}</span>
    </div>`).join('') || '<div style="color:var(--muted);font-size:11px">Henüz veri yok</div>';
}

// ── API CALLS ──────────────────────────────────────────────────────────────────
async function doLogin() {
  sessionId = document.getElementById('session-id').value.trim() || 'default';
  window.location.href = `/auth/login?session_id=${encodeURIComponent(sessionId)}`;
}

async function startStream() {
  const url = document.getElementById('video-url').value.trim();
  if (!url) { showAlert('Hata', 'YouTube URL giriniz'); return; }
  sessionId = document.getElementById('session-id').value.trim() || 'default';
  setLoading(true);
  try {
    const r = await fetch('/api/start', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({video_url:url, session_id:sessionId})});
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail||'Hata');
    addLog('STREAM', `Yayın başlatıldı — video:${d.video_id} chat:${d.live_chat_id}`, false);
    isRunning = true; updateRunningUI();
  } catch(e) { showAlert('Hata', e.message); addLog('ERR', e.message, true); }
  setLoading(false);
}

async function stopStream() {
  await fetch('/api/stop', {method:'POST'});
  isRunning = false; updateRunningUI();
  addLog('STREAM', 'Yayın durduruldu', false);
}

async function injectTest() {
  setLoading(true);
  try {
    const r = await fetch('/api/inject_test', {method:'POST'});
    const d = await r.json();
    addLog('TEST', `${d.injected} mesaj inject edildi — ${d.clusters} küme`, false);
    showAlert('✅ Test verisi inject edildi', `${d.injected} mesaj · ${d.clusters} küme`);
    await refreshClusters();
  } catch(e) { showAlert('Hata', e.message); }
  setLoading(false);
}

async function refreshClusters() {
  const r = await fetch('/api/clusters');
  const d = await r.json();
  allClusters = d.clusters;
  renderClusters(); renderReportQueue(); updateChartPanel();
}

async function massReport() {
  if (!selectedClusters.size) { showAlert('Uyarı', 'Önce küme seçin'); return; }
  sessionId = document.getElementById('session-id').value.trim() || 'default';
  const autoDelete = document.getElementById('auto-delete').checked;
  setLoading(true);
  try {
    const r = await fetch('/api/report', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({cluster_ids:[...selectedClusters], session_id:sessionId, auto_delete:autoDelete})});
    const d = await r.json();
    let total = 0;
    for (const res of d.results) {
      total += res.accounts_reported;
      addLog('REPORT', `${res.cluster_id} → ${res.accounts_reported} hesap raporlandı, ${res.messages_deleted} mesaj silindi`, false);
      addLog('YT-REPORT', `${res.cluster_id} → ${ytReportInfo(res.youtube_report)}`, !res.report_sent);
    }
    const okCount = d.results.filter(x => x.report_sent).length;
    showAlert('✅ Raporlandı', `${d.results.length} küme · ${total} hesap işlendi · YouTube onay: ${okCount}/${d.results.length}`);
    selectedClusters.clear(); await refreshClusters();
  } catch(e) { showAlert('Hata', e.message); }
  setLoading(false);
}

async function simulateReport() {
  if (!selectedClusters.size) { showAlert('Uyarı', 'Önce küme seçin'); return; }
  const r = await fetch('/api/report_bulk_simulate', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({cluster_ids:[...selectedClusters]})});
  const d = await r.json();
  let total = 0;
  for (const res of d.results) { total += res.accounts_reported; addLog('SIM-REPORT', `[SIM] ${res.cluster_id} → ${res.accounts_reported} hesap`, false); }
  showAlert('✅ Simülasyon tamamlandı', `${d.results.length} küme · ${total} hesap (demo)`);
  selectedClusters.clear(); await refreshClusters();
}

async function quickReport(cid) {
  sessionId = document.getElementById('session-id').value.trim() || 'default';
  setLoading(true);
  try {
    const r = await fetch('/api/report', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({cluster_ids:[cid], session_id:sessionId, auto_delete:false})});
    const d = await r.json();
    addLog('REPORT', `${cid} → ${d.results[0]?.accounts_reported} hesap`, false);
    addLog('YT-REPORT', `${cid} → ${ytReportInfo(d.results[0]?.youtube_report)}`, !(d.results[0]?.report_sent));
    const okTxt = d.results[0]?.report_sent ? 'YouTube tarafından onaylandı' : 'YouTube onayı alınamadı';
    showAlert('✅ Raporlandı', `${cid} bildirimi gönderildi · ${okTxt}`);
    await refreshClusters();
  } catch(e) { showAlert('Hata', e.message); }
  setLoading(false);
}

async function simulateSingle(cid) {
  await fetch('/api/report_bulk_simulate', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({cluster_ids:[cid]})});
  showAlert('✅ Simüle edildi', `${cid} raporlandı (demo)`);
  await refreshClusters();
}

// ── HELPERS ───────────────────────────────────────────────────────────────────
function esc(s='') { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function setLoading(on) { document.getElementById('loading-bar').className = 'loading-bar'+(on?' active':''); }

let alertTO = null;
function showAlert(title, body) {
  document.querySelector('.alert-toast')?.remove();
  const d = document.createElement('div');
  d.className = 'alert-toast';
  d.innerHTML = `<div class="alert-title">${title}</div><div class="alert-body">${body}</div>`;
  document.body.appendChild(d);
  clearTimeout(alertTO);
  alertTO = setTimeout(() => d.remove(), 5000);
}

function addLog(event, val, isAlert) {
  const ts = new Date().toLocaleTimeString('tr-TR',{hour12:false});
  const c  = document.getElementById('log-container');
  const d  = document.createElement('div');
  d.className = 'log-line'+(isAlert?' alert':'');
  d.innerHTML = `<span class="ts">[${ts}]</span> <span class="evt">${event}</span> <span class="val">${esc(val)}</span>`;
  c.appendChild(d);
  c.scrollTop = c.scrollHeight;
}

function ytReportInfo(meta) {
  if (!meta) return 'YouTube yanıt bilgisi yok';
  if (!meta.attempted) return `YouTube çağrısı yapılmadı (${meta.error || 'neden belirtilmedi'})`;
  const status = meta.status_code ?? 'n/a';
  const okTxt = meta.report_sent ? 'ONAYLANDI' : 'BAŞARISIZ';
  const excerpt = meta.response_excerpt ? ` | yanıt: ${String(meta.response_excerpt).substring(0,100)}` : '';
  return `YouTube reportAbuse ${okTxt} | HTTP ${status}${excerpt}`;
}

// ── USER PANEL ─────────────────────────────────────────────────────────────────
function renderUsers() {
  const el  = document.getElementById('user-list');
  const cnt = document.getElementById('user-count');
  const bar = document.getElementById('all-stop-bar');
  const aCount = document.getElementById('active-reporter-count');

  if (!liveUsers.size) {
    el.innerHTML = `<div class="empty-state"><div class="icon">👥</div><p>Henüz kullanıcı görülmedi.<br>Mesajlar gelince liste burada oluşur.</p></div>`;
    cnt.textContent = '0 kullanıcı';
    bar.style.display = 'none';
    return;
  }

  const activeCount = userReports.size;
  cnt.textContent = liveUsers.size + ' kullanıcı';
  bar.style.display = activeCount > 0 ? 'flex' : 'none';
  if (aCount) aCount.textContent = activeCount;

  const sorted = [...liveUsers.values()].sort((a,b) => b.lastSeen - a.lastSeen);
  el.innerHTML = sorted.map(u => {
    const name = u.displayName;
    const state = userReports.get(name);
    const isActive = !!state;
    const hateRatio = u.msgCount > 0 ? (u.hateCount / u.msgCount) : 0;
    const dangerColor = hateRatio > 0.5 ? 'var(--accent)' : hateRatio > 0.2 ? 'var(--orange)' : 'var(--muted)';
    const safeId = encodeURIComponent(name).replace(/%/g,'_');
    const suggested = u.suggestedReason || 'spam';
    const suggestedLabel = REPORT_REASONS.find(r => r.id === suggested)?.label || suggested;
    const topHitCount = (u.categoryHits && u.categoryHits[suggested]) ? u.categoryHits[suggested] : 0;
    const reasonOpts = REPORT_REASONS.map(r =>
      `<option value="${r.id}" ${r.id === suggested ? 'selected' : ''}>${r.label}</option>`).join('');
    return `
    <div class="user-row ${isActive ? 'reporting' : ''}" id="urow-${safeId}">
      <div class="user-info">
        <div class="user-name">@${esc(name)}</div>
        <div class="user-meta">
          <span>${u.msgCount} mesaj</span>
          ${u.hateCount > 0 ? `<span style="color:${dangerColor}"> · ⚠️ ${u.hateCount} nefret</span>` : ''}
        </div>
        <div class="user-meta" style="margin-top:4px;">
          <span>🤖 Öneri: <b style="color:var(--yellow)">${esc(suggestedLabel)}</b>${topHitCount ? ` (${topHitCount} eşleşme)` : ''}</span>
        </div>
        ${isActive ? `<div class="report-fire">✅ ${state.count} KEZ RAPORLANDI (tek-sefer)</div>` : ''}
      </div>
      <div class="user-actions">
        <select class="reason-select" id="reason-${safeId}" ${isActive ? 'disabled' : ''}>
          ${reasonOpts}
        </select>
        ${isActive
          ? `<button class="btn btn-sm btn-rstop" onclick="stopUserReport(${JSON.stringify(name)})">■ Sıfırla</button>`
          : `<button class="btn btn-sm btn-rstart" onclick="startUserReport(${JSON.stringify(name)})">🚨 Tek Rapor</button>`
        }
      </div>
    </div>`;
  }).join('');
}

function inferReasonFromTexts(texts) {
  const hits = {};
  for (const reason of Object.keys(NLP_REASON_PATTERNS)) hits[reason] = 0;
  for (const text of texts) {
    for (const [reason, patterns] of Object.entries(NLP_REASON_PATTERNS)) {
      for (const pattern of patterns) {
        if (pattern.test(text || '')) hits[reason]++;
      }
    }
  }
  let bestReason = 'spam';
  let bestScore = -1;
  for (const [reason, score] of Object.entries(hits)) {
    if (score > bestScore) {
      bestReason = reason;
      bestScore = score;
    }
  }
  return { reason: bestReason, hits };
}

async function startUserReport(name) {
  if (userReports.has(name)) return;
  const safeId = encodeURIComponent(name).replace(/%/g,'_');
  const reasonEl = document.getElementById('reason-' + safeId);
  const user = liveUsers.get(name);
  const suggested = user?.suggestedReason || 'spam';
  const reason = reasonEl ? reasonEl.value : suggested;
  const reasonLabel = REPORT_REASONS.find(r => r.id === reason)?.label || reason;

  const state = { count: 0, reason, lastAt: Date.now() };
  userReports.set(name, state);

  state.count++;
  let payload = null;
  try {
    const res = await fetch('/api/report_user_simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ display_name: name, reason, session_id: sessionId })
    });
    payload = await res.json();
    addLog('YT-REPORT', `@${name} → ${ytReportInfo(payload.youtube_report)}`, !payload.report_sent);
  } catch(_) {}

  renderUsers();
  addLog('USER-REPORT', `Gönderildi → @${name} | Neden: ${reasonLabel}`, true);
  if (payload?.report_sent) {
    showAlert('🚨 Rapor Gönderildi', `@${name} için YouTube bildirimi onaylandı (HTTP ${payload.youtube_report?.status_code ?? 'n/a'})`);
  } else {
    showAlert('⚠️ Rapor Durumu', `@${name} için YouTube onayı alınamadı; Log sekmesinde detay mevcut`);
  }
}

function stopUserReport(name) {
  const state = userReports.get(name);
  if (!state) return;
  userReports.delete(name);
  addLog('USER-REPORT', `Sıfırlandı → @${name} | Toplam: ${state.count} rapor`, false);
  renderUsers();
}

function stopAllReports() {
  for (const [name, state] of userReports.entries()) {
    addLog('USER-REPORT', `Sıfırlandı → @${name} | Toplam: ${state.count} rapor`, false);
  }
  userReports.clear();
  renderUsers();
  showAlert('■ Tüm Raporlamalar Durduruldu', `Aktif raporlama oturumu sona erdi`);
}

// ── INIT ──────────────────────────────────────────────────────────────────────
connectWS();
setInterval(refreshClusters, 5000);
addLog('UI', 'Dashboard başlatıldı — WebSocket bekleniyor', false);
</script>
</body>
</html>"""


# ─── CONFIG ────────────────────────────────────────────────────────────────────
GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
REDIRECT_URI         = os.getenv("REDIRECT_URI", "http://localhost:8000/auth/callback")
PORT                 = int(os.getenv("PORT", "8000"))

SCOPES = [
    "https://www.googleapis.com/auth/youtube.force-ssl",
    "https://www.googleapis.com/auth/youtube.readonly",
]

RISK_WEIGHTS = {
    "hate_score":       0.28,
    "semantic_cluster": 0.20,
    "burst_sync":       0.17,
    "style_fp":         0.13,
    "account_behavior": 0.10,
    "graph_cluster":    0.12,
}

POLL_INTERVAL = 1.0   # saniye — YouTube API kendi aralığını döndürür, bu alt sınır
BURST_WINDOW  = 10.0
SIM_THRESHOLD = 0.75
FLOOD_BURST   = 5


# ─── HATE PATTERNS ─────────────────────────────────────────────────────────────
ANTISEMITIC_PATTERNS = [
    r"\bjew[s]?\b", r"\bjewish\b", r"\bkike[s]?\b", r"\bzhid\b", r"\bjude\b",
    r"\bholocaust\s*deni", r"\bgas\s*the\s*jew", r"\bsix\s*million\s*lie",
    r"\bzionist\s*pig", r"\bworld\s*jewish\s*conspir", r"\bheil\b",
    r"\bstürmer\b", r"\bprotokoll[ei]\b", r"\bsubhuman.*jew", r"\bjew.*subhuman",
    r"\brat.*jew", r"\bjew.*rat", r"\b14\s*words\b", r"\b88\b",
    r"\bwhite\s*power\b", r"\bnazi\b", r"\breich\b.*\bsieg\b",
]
HATE_PATTERNS = ANTISEMITIC_PATTERNS + [
    r"\bn[i1!|]+g+[e3]+r[s]?\b", r"\bfagg?[o0]+t[s]?\b", r"\bk[y]+k[e]+\b",
    r"\bchink[s]?\b", r"\bsp[i1]+c[k]?[s]?\b", r"\bwetback\b",
    r"\bk[i1!]+ll\s+(all|the)\s+(jew|muslim|black|white)",
    r"\bde[a4]th\s+to\b", r"\bg[e3]n[o0]c[i1]d[e3]\b",
    r"\bheil\s+\w+", r"\bblut\s+und\s+boden\b",
]
COMPILED_HATE = [re.compile(p, re.IGNORECASE) for p in HATE_PATTERNS]

DOGWHISTLES = {
    "1488": 1.0, "rwds": 0.9, "hh": 0.8, "zog": 0.9, "jtb": 0.9,
    "groyper": 0.7, "baste": 0.5, "red pill": 0.4, "clown world": 0.8,
    "great replacement": 0.95, "white genocide": 1.0, "kalergi": 0.9,
}

HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "і": "i", "о": "o", "р": "p", "с": "c",
    "у": "y", "х": "x", "ё": "e", "ι": "i", "ο": "o", "α": "a",
    "1": "l", "0": "o", "3": "e", "4": "a", "5": "s", "@": "a",
    "!": "i", "|": "l", "$": "s",
})


# ─── DATA STRUCTURES ──────────────────────────────────────────────────────────
@dataclass
class ParsedMessage:
    id: str
    channel_id: str
    display_name: str
    text: str
    normalized: str
    published_at: float
    exact_hash: str
    sim_hash: int
    embedding_tokens: list
    hate_score: float
    hate_categories: list
    account_age_days: int
    account_flags: list


@dataclass
class Cluster:
    cluster_id: str
    messages: list       = field(default_factory=list)
    accounts: set        = field(default_factory=set)
    phrases: dict        = field(default_factory=dict)
    first_seen: float    = field(default_factory=time.time)
    last_seen: float     = field(default_factory=time.time)
    risk_level: str      = "LOW"
    risk_score: float    = 0.0
    reported: bool       = False
    hate_categories: list = field(default_factory=list)


# ─── STATE ────────────────────────────────────────────────────────────────────
class EngineState:
    def __init__(self):
        self.oauth_tokens: dict         = {}
        self.live_chat_id: Optional[str] = None
        self.next_page_token: Optional[str] = None
        self.running: bool              = False
        self.video_id: Optional[str]   = None

        self.all_messages: deque        = deque(maxlen=5000)
        self.hash_index: dict           = defaultdict(list)
        self.sim_buckets: dict          = defaultdict(list)
        self.account_messages: dict     = defaultdict(list)
        self.account_timeline: dict     = defaultdict(list)

        self.clusters: dict             = {}
        self.message_to_cluster: dict   = {}
        self.ws_clients: list           = []

        self.total_scanned: int         = 0
        self.total_hate: int            = 0
        self.total_reported: int        = 0
        self.start_time: float          = time.time()

    def reset_stream(self):
        self.live_chat_id = None
        self.next_page_token = None
        self.all_messages.clear()
        self.hash_index.clear()
        self.sim_buckets.clear()
        self.account_messages.clear()
        self.account_timeline.clear()
        self.clusters.clear()
        self.message_to_cluster.clear()
        self.total_scanned = 0
        self.total_hate = 0
        self.total_reported = 0
        self.start_time = time.time()


STATE = EngineState()


# ─── NORMALIZATION ENGINE ──────────────────────────────────────────────────────
def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    t = t.translate(HOMOGLYPHS)
    t = t.lower()
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"[^\w\s\-]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# ─── HATE DETECTOR ────────────────────────────────────────────────────────────
def detect_hate(text: str, normalized: str) -> tuple:
    score = 0.0
    categories = []

    for pat in COMPILED_HATE:
        if pat.search(normalized):
            score += 0.4
            cat = pat.pattern[:20]
            if cat not in categories:
                categories.append(cat)

    tokens = normalized.split()
    for token in tokens:
        if token in DOGWHISTLES:
            score += DOGWHISTLES[token]
            categories.append(f"dogwhistle:{token}")

    if "(((" in text or ")))" in text:
        score += 0.9
        categories.append("antisemit:triple_paren")

    if len(tokens) > 3:
        bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)]
        if max((bigrams.count(p) for p in set(bigrams)), default=0) > 2:
            score += 0.2

    return min(score, 1.0), categories


# ─── SIMHASH ──────────────────────────────────────────────────────────────────
def simhash(text: str, bits: int = 64) -> int:
    v = [0] * bits
    tokens = text.split()
    if not tokens:
        return 0
    for token in tokens:
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        for i in range(bits):
            v[i] += 1 if (h & (1 << i)) else -1
    fp = 0
    for i in range(bits):
        if v[i] > 0:
            fp |= 1 << i
    return fp


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def sim_similarity(a: int, b: int, bits: int = 64) -> float:
    return 1.0 - hamming_distance(a, b) / bits


def simhash_bucket(sh: int, bands: int = 8) -> list:
    bpb = 64 // bands
    return [f"b{i}:{(sh >> (i*bpb)) & ((1 << bpb)-1)}" for i in range(bands)]


# ─── ACCOUNT BEHAVIOR ─────────────────────────────────────────────────────────
def account_behavior_score(channel_id: str, msg_time: float, account_age_days: int) -> tuple:
    score = 0.0
    flags = []
    timeline = STATE.account_timeline[channel_id]
    timeline.append(msg_time)

    if account_age_days < 7:
        score += 0.5; flags.append("new_account_<7d")
    elif account_age_days < 30:
        score += 0.3; flags.append("new_account_<30d")

    recent = [t for t in timeline if msg_time - t < BURST_WINDOW]
    if len(recent) >= FLOOD_BURST:
        score += 0.6; flags.append(f"flood_burst_{len(recent)}_in_{BURST_WINDOW}s")

    return min(score, 1.0), flags


# ─── BURST ENGINE ─────────────────────────────────────────────────────────────
def detect_swarm_burst(msg_time: float) -> float:
    recent = [m for m in STATE.all_messages if msg_time - m.published_at < 5.0]
    if len(recent) < 3:
        return 0.0
    unique = len(set(m.channel_id for m in recent))
    hate_ratio = sum(1 for m in recent if m.hate_score > 0.3) / max(len(recent), 1)
    return min(unique / 20.0, 1.0) * hate_ratio


# ─── CLUSTERING ENGINE ────────────────────────────────────────────────────────
def find_or_create_cluster(msg: ParsedMessage) -> str:
    if msg.exact_hash in STATE.hash_index and STATE.hash_index[msg.exact_hash]:
        cid = STATE.message_to_cluster.get(STATE.hash_index[msg.exact_hash][0].id)
        if cid:
            return cid

    for bucket in simhash_bucket(msg.sim_hash):
        for candidate in STATE.sim_buckets.get(bucket, []):
            if sim_similarity(msg.sim_hash, candidate.sim_hash) >= SIM_THRESHOLD:
                cid = STATE.message_to_cluster.get(candidate.id)
                if cid:
                    return cid

    for am in STATE.account_messages.get(msg.channel_id, [])[-5:]:
        cid = STATE.message_to_cluster.get(am.id)
        if cid and STATE.clusters[cid].risk_score > 0.4:
            return cid

    cid = f"CLU-{hashlib.md5(f'{msg.exact_hash}{time.time()}'.encode()).hexdigest()[:8].upper()}"
    STATE.clusters[cid] = Cluster(cluster_id=cid)
    return cid


def update_cluster(cid: str, msg: ParsedMessage, risk_score: float):
    c = STATE.clusters[cid]
    c.messages.append({
        "id": msg.id, "channel_id": msg.channel_id,
        "display_name": msg.display_name, "text": msg.text,
        "published_at": msg.published_at, "hate_score": msg.hate_score,
    })
    c.accounts.add(msg.channel_id)
    c.last_seen  = msg.published_at
    c.risk_score = max(c.risk_score, risk_score)

    words = msg.normalized.split()
    for i in range(len(words) - 1):
        p = f"{words[i]} {words[i+1]}"
        c.phrases[p] = c.phrases.get(p, 0) + 1

    for cat in msg.hate_categories:
        if cat not in c.hate_categories:
            c.hate_categories.append(cat)

    if risk_score >= 0.85:   c.risk_level = "SWARM ATTACK"
    elif risk_score >= 0.70: c.risk_level = "COORDINATED RAID"
    elif risk_score >= 0.50: c.risk_level = "HIGH"
    elif risk_score >= 0.30: c.risk_level = "MEDIUM"
    else:                    c.risk_level = "LOW"

    STATE.message_to_cluster[msg.id] = cid


# ─── RISK SCORER ──────────────────────────────────────────────────────────────
def compute_risk(hate_score, burst_sync, account_behavior, sim_cluster=0.0, graph_cluster=0.0) -> float:
    return min(
        hate_score      * RISK_WEIGHTS["hate_score"]
        + sim_cluster   * RISK_WEIGHTS["semantic_cluster"]
        + burst_sync    * RISK_WEIGHTS["burst_sync"]
        + account_behavior * RISK_WEIGHTS["account_behavior"]
        + graph_cluster * RISK_WEIGHTS["graph_cluster"],
        1.0
    )


# ─── MESSAGE PROCESSOR ────────────────────────────────────────────────────────
async def process_message(raw: dict) -> Optional[ParsedMessage]:
    snippet = raw.get("snippet", {})
    author  = raw.get("authorDetails", {})
    text = snippet.get("displayMessage", "").strip()
    if not text:
        return None

    channel_id   = author.get("channelId", "UNKNOWN")
    display_name = author.get("displayName", "?")
    msg_id       = raw.get("id", hashlib.md5(text.encode()).hexdigest())
    published_str = snippet.get("publishedAt", "")

    try:
        published_at = datetime.fromisoformat(published_str.replace("Z", "+00:00")).timestamp()
    except Exception:
        published_at = time.time()

    norm       = normalize(text)
    exact_hash = hashlib.sha256(norm.encode()).hexdigest()[:16]
    sh         = simhash(norm)
    hate_score, hate_cats = detect_hate(text, norm)
    acct_score, acct_flags = account_behavior_score(channel_id, published_at, 999)
    burst_sync = detect_swarm_burst(published_at)

    msg = ParsedMessage(
        id=msg_id, channel_id=channel_id, display_name=display_name,
        text=text, normalized=norm, published_at=published_at,
        exact_hash=exact_hash, sim_hash=sh,
        embedding_tokens=norm.split()[:20],
        hate_score=hate_score, hate_categories=hate_cats,
        account_age_days=999, account_flags=acct_flags,
    )

    STATE.all_messages.append(msg)
    STATE.hash_index[exact_hash].append(msg)
    STATE.account_messages[channel_id].append(msg)
    for bucket in simhash_bucket(sh):
        STATE.sim_buckets[bucket].append(msg)

    STATE.total_scanned += 1
    if hate_score > 0.2:
        STATE.total_hate += 1

    sim_cluster = min(len(STATE.hash_index[exact_hash]) / 10.0, 1.0) if len(STATE.hash_index[exact_hash]) > 1 else 0.0
    risk = compute_risk(hate_score, burst_sync, acct_score, sim_cluster)

    if risk > 0.15 or hate_score > 0.2:
        cid = find_or_create_cluster(msg)
        update_cluster(cid, msg, risk)

    return msg


# ─── YOUTUBE API CLIENT ───────────────────────────────────────────────────────
async def yt_get(endpoint: str, params: dict, access_token: str) -> dict:
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(
            f"https://www.googleapis.com/youtube/v3/{endpoint}",
            params=params,
            headers={"Authorization": f"Bearer {access_token}"}
        )
        r.raise_for_status()
        return r.json()


async def resolve_live_chat_id(video_id: str, access_token: str) -> str:
    data = await yt_get("videos", {"part": "liveStreamingDetails", "id": video_id}, access_token)
    items = data.get("items", [])
    if not items:
        raise ValueError(f"Video {video_id} not found")
    lcd = items[0].get("liveStreamingDetails", {}).get("activeLiveChatId")
    if not lcd:
        raise ValueError("No active live chat for this video")
    return lcd


async def fetch_chat_page(live_chat_id: str, page_token: Optional[str], access_token: str) -> dict:
    params = {"part": "snippet,authorDetails", "liveChatId": live_chat_id, "maxResults": 2000}
    if page_token:
        params["pageToken"] = page_token
    return await yt_get("liveChat/messages", params, access_token)


async def report_abuse(video_id: str, access_token: str, reason_id: str = "hateSpeech") -> bool:
    body = {
        "videoId": video_id, "reasonId": reason_id, "secondaryReasonId": "",
        "comments": "Coordinated hate speech / antisemitic swarm attack detected by automated moderation engine.",
        "language": "en",
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(
            "https://www.googleapis.com/youtube/v3/videos/reportAbuse",
            json=body,
            headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
            params={"part": ""},
        )
        return r.status_code in (200, 204)


async def report_abuse_detailed(
    video_id: str,
    access_token: Optional[str],
    *,
    reason_id: str = "hateSpeech",
    secondary_reason_id: str = "",
    comments: str = "Automated moderation report request.",
    language: str = "en",
) -> dict:
    if not access_token:
        return {
            "attempted": False,
            "report_sent": False,
            "status_code": None,
            "response_excerpt": "No access token",
            "error": "missing_access_token",
        }
    if not video_id:
        return {
            "attempted": False,
            "report_sent": False,
            "status_code": None,
            "response_excerpt": "No active video",
            "error": "missing_video_id",
        }

    payload = {
        "videoId": video_id,
        "reasonId": reason_id,
        "secondaryReasonId": secondary_reason_id,
        "comments": comments,
        "language": language,
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                "https://www.googleapis.com/youtube/v3/videos/reportAbuse",
                params={"videoId": video_id},
                json=payload,
                headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
            )
        raw = (r.text or "").strip().replace("\n", " ")
        excerpt = raw[:180] if raw else "empty-response-body"
        return {
            "attempted": True,
            "report_sent": r.status_code in (200, 204),
            "status_code": r.status_code,
            "response_excerpt": excerpt,
            "error": None if r.status_code in (200, 204) else "youtube_non_success_status",
        }
    except Exception as e:
        return {
            "attempted": True,
            "report_sent": False,
            "status_code": None,
            "response_excerpt": "",
            "error": str(e),
        }


async def delete_message(msg_id: str, access_token: str) -> bool:
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.delete(
            "https://www.googleapis.com/youtube/v3/liveChat/messages",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"id": msg_id},
        )
        return r.status_code == 204


# ─── POLLING LOOP ─────────────────────────────────────────────────────────────
async def polling_loop(session_id: str):
    token_data = STATE.oauth_tokens.get(session_id, {})
    access_token = token_data.get("access_token")
    if not access_token or not STATE.live_chat_id:
        return

    STATE.running = True
    poll_interval = POLL_INTERVAL

    while STATE.running:
        try:
            # Token'ı her döngüde taze al (refresh sonrası güncel olsun)
            token_data = STATE.oauth_tokens.get(session_id, {})
            access_token = token_data.get("access_token")
            if not access_token:
                print("[POLL] Erişim token'ı yok, durduruluyor.")
                STATE.running = False
                break

            data = await fetch_chat_page(STATE.live_chat_id, STATE.next_page_token, access_token)
            STATE.next_page_token = data.get("nextPageToken")
            # YouTube önerilen aralığını döndürür ama biz 1.5s'de sabitliyoruz
            poll_interval = 1.5
            for item in data.get("items", []):
                await process_message(item)
            await broadcast_update()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            print(f"[POLL ERROR] HTTP {status}: {e.response.text[:200]}")
            if status == 401:
                # Token süresi dolmuş — refresh dene
                refresh_token = STATE.oauth_tokens.get(session_id, {}).get("refresh_token")
                if refresh_token and GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
                    try:
                        async with httpx.AsyncClient() as client:
                            r = await client.post("https://oauth2.googleapis.com/token", data={
                                "client_id": GOOGLE_CLIENT_ID,
                                "client_secret": GOOGLE_CLIENT_SECRET,
                                "refresh_token": refresh_token,
                                "grant_type": "refresh_token",
                            })
                            new_tokens = r.json()
                            if "access_token" in new_tokens:
                                STATE.oauth_tokens[session_id].update(new_tokens)
                                print("[POLL] Token yenilendi.")
                                continue
                    except Exception as re:
                        print(f"[POLL] Token yenileme başarısız: {re}")
                STATE.running = False
                break
            elif status == 403:
                print("[POLL] 403 Erişim reddedildi — kota veya izin sorunu.")
                STATE.running = False
                break
            await asyncio.sleep(poll_interval)
        except Exception as e:
            print(f"[POLL ERROR] {e}")
            await asyncio.sleep(poll_interval)


# ─── WEBSOCKET BROADCAST ──────────────────────────────────────────────────────
def serialize_cluster(c: Cluster) -> dict:
    return {
        "cluster_id":    c.cluster_id,
        "risk_level":    c.risk_level,
        "risk_score":    round(c.risk_score, 3),
        "account_count": len(c.accounts),
        "message_count": len(c.messages),
        "hate_categories": c.hate_categories[:5],
        "top_phrases":   sorted(c.phrases.items(), key=lambda x: x[1], reverse=True)[:5],
        "sample_messages": c.messages[-10:],
        "reported":      c.reported,
        "first_seen":    c.first_seen,
        "last_seen":     c.last_seen,
        "accounts":      list(c.accounts)[:20],
    }


async def broadcast_update():
    if not STATE.ws_clients:
        return
    top_threats = sorted(
        [c for c in STATE.clusters.values() if c.hate_categories],
        key=lambda c: c.risk_score, reverse=True
    )[:50]
    recent = [
        {
            "display_name": m.display_name,
            "text": m.text,
            "hate_score": round(m.hate_score, 3),
            "hate_categories": m.hate_categories,
        }
        for m in list(STATE.all_messages)[-30:]
    ]
    payload = {
        "type": "update",
        "stats": {
            "total_scanned": STATE.total_scanned,
            "total_hate":    STATE.total_hate,
            "total_reported": STATE.total_reported,
            "uptime":        int(time.time() - STATE.start_time),
            "cluster_count": len(STATE.clusters),
            "running":       STATE.running,
        },
        "threats": [serialize_cluster(c) for c in top_threats],
        "recent_messages": recent,
    }
    dead = []
    for ws in STATE.ws_clients:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        STATE.ws_clients.remove(ws)


# ─── APP + ROUTES ─────────────────────────────────────────────────────────────
app = FastAPI(title="Swarm Moderation Engine")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
async def root():
    return HTMLResponse(FRONTEND_HTML)


@app.get("/auth/login")
async def auth_login(session_id: str = "default"):
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(400, "GOOGLE_CLIENT_ID not configured. Set env var.")
    scope = " ".join(SCOPES)
    url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        f"?client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&response_type=code"
        f"&scope={scope}"
        f"&access_type=offline&prompt=consent"
        f"&state={session_id}"
    )
    return RedirectResponse(url)


@app.get("/auth/callback")
async def auth_callback(code: str, state: str = "default"):
    async with httpx.AsyncClient() as client:
        r = await client.post("https://oauth2.googleapis.com/token", data={
            "code": code, "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": REDIRECT_URI, "grant_type": "authorization_code",
        })
        STATE.oauth_tokens[state] = r.json()
    return RedirectResponse(f"/?session={state}&auth=ok")


@app.post("/api/start")
async def start_stream(req: Request):
    body = await req.json()
    video_url  = body.get("video_url", "")
    session_id = body.get("session_id", "default")
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_\-]{11})", video_url)
    if not m:
        raise HTTPException(400, "Invalid YouTube URL")
    video_id = m.group(1)
    token_data = STATE.oauth_tokens.get(session_id, {})
    access_token = token_data.get("access_token")
    if not access_token:
        raise HTTPException(401, "Not authenticated. Login via /auth/login first.")
    STATE.reset_stream()
    STATE.video_id = video_id
    try:
        STATE.live_chat_id = await resolve_live_chat_id(video_id, access_token)
    except Exception as e:
        raise HTTPException(400, str(e))
    asyncio.create_task(polling_loop(session_id))
    return {"ok": True, "video_id": video_id, "live_chat_id": STATE.live_chat_id}


@app.post("/api/stop")
async def stop_stream():
    STATE.running = False
    return {"ok": True}


@app.get("/api/clusters")
async def get_clusters():
    sorted_clusters = sorted(STATE.clusters.values(), key=lambda c: c.risk_score, reverse=True)
    return {"clusters": [serialize_cluster(c) for c in sorted_clusters[:100]], "total": len(STATE.clusters)}


@app.post("/api/report")
async def mass_report(req: Request):
    body         = await req.json()
    cluster_ids  = body.get("cluster_ids", [])
    session_id   = body.get("session_id", "default")
    auto_delete  = body.get("auto_delete", False)
    access_token = STATE.oauth_tokens.get(session_id, {}).get("access_token")
    if not access_token:
        raise HTTPException(401, "Not authenticated")

    results = []
    for cid in cluster_ids:
        c = STATE.clusters.get(cid)
        if not c:
            continue
        report_meta = await report_abuse_detailed(
            STATE.video_id or "",
            access_token,
            comments="Cluster-level moderation report from SwarmMod.",
            language="en",
        )
        deleted = 0
        if auto_delete:
            for msg_info in c.messages[-50:]:
                try:
                    if await delete_message(msg_info["id"], access_token):
                        deleted += 1
                except Exception:
                    pass
        c.reported = True
        STATE.total_reported += len(c.accounts)
        results.append({
            "cluster_id": cid, "accounts_reported": len(c.accounts),
            "messages_deleted": deleted, "report_sent": report_meta["report_sent"],
            "youtube_report": report_meta,
        })
    await broadcast_update()
    return {"ok": True, "results": results}


@app.post("/api/report_bulk_simulate")
async def simulate_report(req: Request):
    body = await req.json()
    results = []
    for cid in body.get("cluster_ids", []):
        c = STATE.clusters.get(cid)
        if c:
            c.reported = True
            STATE.total_reported += len(c.accounts)
            results.append({
                "cluster_id": cid, "accounts_reported": len(c.accounts),
                "messages_deleted": len(c.messages), "report_sent": True, "simulated": True,
            })
    await broadcast_update()
    return {"ok": True, "results": results}


@app.post("/api/report_user_simulate")
async def report_user_simulate(req: Request):
    """
    Bireysel kullanıcı rapor simülasyonu.
    Gerçek OAuth bağlantısı varsa YouTube videos.reportAbuse API'sine çağrı yapar.
    """
    body         = await req.json()
    display_name = body.get("display_name", "unknown")
    reason       = body.get("reason", "spam")
    session_id   = body.get("session_id", "default")

    REASON_MAP = {
        "spam":     ("S", "30"),
        "sexual":   ("V", "1"),
        "child":    ("Y", "0"),
        "hate":     ("H", "1"),
        "terror":   ("T", "0"),
        "harass":   ("A", "0"),
        "selfharm": ("M", "0"),
        "misinfo":  ("R", "0"),
    }
    reason_id, secondary_id = REASON_MAP.get(reason, ("S", "30"))

    access_token = STATE.oauth_tokens.get(session_id, {}).get("access_token")
    report_meta = await report_abuse_detailed(
        STATE.video_id or "",
        access_token,
        reason_id=reason_id,
        secondary_reason_id=secondary_id,
        comments=f"Reported user: {display_name}",
        language="tr",
    )

    STATE.total_reported += 1
    print(
        f"[USER-REPORT] {'REAL' if report_meta['report_sent'] else 'SIM'} "
        f"| user={display_name} reason={reason} status={report_meta.get('status_code')}"
    )
    return {
        "ok": True,
        "display_name": display_name,
        "reason": reason,
        "report_sent": report_meta["report_sent"],
        "simulated": not report_meta["report_sent"],
        "youtube_report": report_meta,
    }


@app.post("/api/inject_test")
async def inject_test_data():
    import random
    test_accounts = [f"UC{hashlib.md5(str(i).encode()).hexdigest()[:20]}" for i in range(15)]
    hate_phrases = [
        "gas the jews now!!", "Jеwwws out of our country!!!",
        "((( they ))) control everything", "1488 white power!!",
        "heil our leader death to all", "kill all the jews",
        "jewish conspiracy exposed", "zog controls the media",
        "ZOG ZOG ZOG 14 words", "Jews did 9/11 proof",
    ]
    now = time.time()
    injected = 0

    for i, acc in enumerate(test_accounts[:10]):
        phrase = hate_phrases[i % len(hate_phrases)]
        await process_message({
            "id": f"fake_{acc}_{i}",
            "snippet": {
                "displayMessage": phrase,
                "publishedAt": datetime.fromtimestamp(now - random.uniform(0, 5), tz=timezone.utc).isoformat(),
            },
            "authorDetails": {"channelId": acc, "displayName": f"TestUser{i}"},
        })
        injected += 1

    for i in range(5):
        acc = test_accounts[10 + i % 5]
        await process_message({
            "id": f"flood_{acc}_{i}",
            "snippet": {
                "displayMessage": "gas the jews now!! (((they))) must go",
                "publishedAt": datetime.fromtimestamp(now - random.uniform(0, 2), tz=timezone.utc).isoformat(),
            },
            "authorDetails": {"channelId": acc, "displayName": f"FloodBot{i}"},
        })
        injected += 1

    await broadcast_update()
    return {"ok": True, "injected": injected, "clusters": len(STATE.clusters)}


@app.get("/api/status")
async def status():
    return {
        "running": STATE.running, "live_chat_id": STATE.live_chat_id,
        "video_id": STATE.video_id, "total_scanned": STATE.total_scanned,
        "total_hate": STATE.total_hate, "total_reported": STATE.total_reported,
        "cluster_count": len(STATE.clusters), "uptime": int(time.time() - STATE.start_time),
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    STATE.ws_clients.append(ws)
    try:
        await broadcast_update()
        while True:
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                await ws.send_json({"type": "ping"})
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        if ws in STATE.ws_clients:
            STATE.ws_clients.remove(ws)


# ─── ENTRYPOINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo = not GOOGLE_CLIENT_ID
    print(f"""
  ╔═══════════════════════════════════════╗
  ║   SWARM MOD ENGINE v2.0               ║
  ║   YouTube Live Hate/Flood Detector    ║
  ╚═══════════════════════════════════════╝
  Mod  : {'DEMO (OAuth yok)' if demo else 'GERÇEK (OAuth aktif)'}
  URL  : http://localhost:{PORT}
""")

uvicorn.run(app, host="0.0.0.0", port=PORT, reload=False)
