# =========================================================================
# === FASA 3: MULTIPLE TIMEFRAME UPLOAD SEKALI ===
# =========================================================================

def launch_fasa3(self):
    """Launch FASA 3 - Upload sekali untuk semua timeframe"""
    st.header("📸 FASA 3: MT5 MULTITIMEFRAME ANALYSIS")
    st.info("UPLOAD SEKALI - AI akan analisa D1, H4, H1, M30, M15, M5, M1")
    
    # UPLOAD SEKALI UNTUK SEMUA TIMEFRAME
    uploaded_files = st.file_uploader(
        "UPLOAD SEMUA SCREENSHOT TIMEFRAME (Pilih multiple files)",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        key="fasa3_multi_upload"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} screenshot diupload!")
        
        # Simpan semua screenshot
        timeframe_mapping = ['D1', 'H4', 'H1', 'M30', 'M15', 'M5', 'M1']
        
        for i, uploaded_file in enumerate(uploaded_files):
            if i < len(timeframe_mapping):  # Pastikan tidak lebih dari 7
                timeframe = timeframe_mapping[i]
                image = Image.open(uploaded_file)
                
                # Simpan ke session state
                if 'uploaded_timeframes' not in st.session_state:
                    st.session_state.uploaded_timeframes = {}
                st.session_state.uploaded_timeframes[timeframe] = image
                
                st.write(f"📁 {timeframe}: {uploaded_file.name}")
        
        # Tunjukkan preview semua screenshot
        st.subheader("📊 PREVIEW SEMUA TIMEFRAME")
        cols = st.columns(7)
        for i, timeframe in enumerate(timeframe_mapping):
            if timeframe in st.session_state.uploaded_timeframes:
                with cols[i]:
                    st.image(st.session_state.uploaded_timeframes[timeframe], 
                            width=80, caption=timeframe)
        
        # BUTANG ANALISA - muncul setelah upload
        if st.button("🔍 AI ANALISA SEMUA TIMEFRAME", type="primary"):
            self.analyze_all_timeframes()

def analyze_all_timeframes(self):
    """Analyze semua timeframe sekaligus"""
    if 'uploaded_timeframes' not in st.session_state:
        st.error("❌ Tiada screenshot untuk dianalisa")
        return
        
    with st.spinner("🧠 AI sedang menganalisa SEMUA timeframe..."):
        analysis_results = {}
        
        for timeframe, screenshot in st.session_state.uploaded_timeframes.items():
            if screenshot is not None:
                # Analyze setiap timeframe
                timeframe_analysis = self.analyze_single_timeframe(screenshot, timeframe)
                analysis_results[timeframe] = timeframe_analysis
                st.write(f"✅ {timeframe}: {len(timeframe_analysis['rules_detected'])} patterns")
        
        # Generate final signal
        final_signal = self.generate_final_signal(analysis_results)
        
        # Tunjukkan results
        self.display_analysis_results(analysis_results, final_signal)
        
        # SIMPAN KNOWLEDGE - PENTING UNTUK AI TIDAK BODOH
        self.save_fasa3_knowledge(analysis_results, final_signal)

def save_fasa3_knowledge(self, analysis_results, final_signal):
    """Simpan knowledge untuk AI terus belajar"""
    # Simpan rule baru berdasarkan analisa
    new_rule = {
        'rule': f'FASA3_MULTITIMEFRAME_{final_signal["trend"]}',
        'confidence': final_signal['confidence'],
        'learned_by': 'fasa3_ai',
        'timestamp': time.time(),
        'session_type': 'multitimeframe_analysis',
        'timeframes_analyzed': list(analysis_results.keys()),
        'total_patterns': sum(len(result['rules_detected']) for result in analysis_results.values())
    }
    
    # Tambah ke memory system - INI YANG BUAT AI TERUS BELAJAR
    success = self.memory.add_sop_rule(new_rule)
    
    if success:
        st.success(f"💾 AI BELAJAR: {new_rule['rule']} (Confidence: {new_rule['confidence']:.1%})")
        
        # Update AI intelligence
        intelligence = self.memory.calculate_ai_intelligence()
        st.metric("🧠 AI INTELLIGENCE UPDATE", f"{intelligence}%")
