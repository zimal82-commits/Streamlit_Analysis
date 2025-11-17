import streamlit as st
import cv2, os, time, pickle
import numpy as np
from PIL import Image
import zipfile
import tempfile

# =========================================================================
# === FIX: MULTIPLE IMAGE UPLOAD SYSTEM ===
# =========================================================================

def setup_multiple_upload():
    """System untuk upload banyak gambar sekali gus"""
    
    st.subheader("📤 UPLOAD BANYAK GAMBAR SEKALIGUS")
    
    # OPTION 1: Multiple file upload
    uploaded_files = st.file_uploader(
        "Pilih BANYAK GAMBAR SEKALIGUS (Ctrl+Click atau Drag Multiple Files)",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        accept_multiple_files=True,  # ← INI YANG PENTING!
        key="multiple_upload"
    )
    
    # OPTION 2: ZIP file upload  
    uploaded_zip = st.file_uploader(
        "Atau UPLOAD ZIP FILE berisi semua gambar",
        type=['zip'],
        key="zip_upload"
    )
    
    all_images = []
    
    # Process multiple files
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} GAMBAR BERJAYA DIUPLOAD!")
        
        for i, uploaded_file in enumerate(uploaded_files):
            # Convert ke OpenCV format
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            all_images.append({
                'name': uploaded_file.name,
                'image': img,
                'data': uploaded_file
            })
            
        # Show preview
        cols = st.columns(4)
        for idx, img_data in enumerate(all_images[:8]):  # Show first 8 only
            with cols[idx % 4]:
                # Convert BGR to RGB untuk display
                img_rgb = cv2.cvtColor(img_data['image'], cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption=img_data['name'], width=100)
        
        if len(all_images) > 8:
            st.info(f"📁 Dan {len(all_images) - 8} gambar lagi...")
    
    # Process ZIP file
    elif uploaded_zip:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract ZIP
            with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Get semua image files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(Path(temp_dir).rglob(ext))
            
            st.success(f"✅ ZIP EXTRACTED: {len(image_files)} GAMBAR DITEMUI!")
            
            for img_path in image_files[:20]:  # Limit untuk performance
                img = cv2.imread(str(img_path))
                if img is not None:
                    all_images.append({
                        'name': img_path.name,
                        'image': img,
                        'path': str(img_path)
                    })
            
            # Show preview
            cols = st.columns(4)
            for idx, img_data in enumerate(all_images[:8]):
                with cols[idx % 4]:
                    img_rgb = cv2.cvtColor(img_data['image'], cv2.COLOR_BGR2RGB)
                    st.image(img_rgb, caption=img_data['name'], width=100)
    
    return all_images

# =========================================================================
# === FASA 1 DENGAN MULTIPLE UPLOAD ===
# =========================================================================

def render_fasa1_enhanced():
    st.header("🎯 FASA 1 - SUPERIOR AI LEARNING (MULTIPLE UPLOAD)")
    
    tab1, tab2 = st.tabs(["📸 UPLOAD BANYAK GAMBAR", "🎬 UPLOAD VIDEO"])
    
    with tab1:
        st.subheader("🖼️ UPLOAD BANYAK GAMBAR SEKALIGUS")
        
        # Gunakan multiple upload system
        all_images = setup_multiple_upload()
        
        if all_images and st.button("🚀 PROCESS ALL IMAGES WITH AI", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_processed = 0
            new_rules_count = 0
            
            for i, img_data in enumerate(all_images):
                status_text.text(f"Processing {i+1}/{len(all_images)}: {img_data['name']}")
                
                # Process setiap image dengan AI
                rules_learned = process_single_image_ai(img_data)
                new_rules_count += len(rules_learned)
                total_processed += 1
                
                progress_bar.progress((i + 1) / len(all_images))
            
            status_text.text(f"✅ PROCESSING COMPLETE!")
            st.success(f"""
            🎉 **AI LEARNING FINISHED!**
            - 📁 Total Images: {len(all_images)}
            - 📚 New Rules Learned: {new_rules_count}
            - 🧠 AI Intelligence Increased!
            """)
    
    with tab2:
        st.subheader("🎬 UPLOAD VIDEO (Single)")
        uploaded_video = st.file_uploader("Pilih video file", type=['mp4', 'avi', 'mov'])
        # ... rest of video code

# =========================================================================
# === FASA 3 DENGAN MULTIPLE ANALYSIS ===
# =========================================================================

def render_fasa3_enhanced():
    st.header("🎯 FASA 3 - AI MARKET ANALYSIS (MULTIPLE CHARTS)")
    
    st.warning("⚠️ UPLOAD BANYAK GAMBAR CHART SEKALIGUS UNTUK BATCH ANALYSIS!")
    
    # Gunakan multiple upload system
    all_charts = setup_multiple_upload()
    
    if all_charts:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📊 BATCH ANALYSIS CONTROL")
            
            analysis_mode = st.radio(
                "PILIH ANALYSIS MODE:",
                ["QUICK ANALYSIS", "DETAILED ANALYSIS", "COMPARE CHARTS"]
            )
            
            if st.button("🔬 ANALYZE ALL CHARTS WITH 100% SOP", type="primary"):
                with st.spinner(f"AI analyzing {len(all_charts)} charts dengan 100% SOP..."):
                    batch_results = analyze_multiple_charts(all_charts, analysis_mode)
                    display_batch_results(batch_results)
        
        with col2:
            st.subheader("📁 UPLOADED CHARTS PREVIEW")
            # Preview akan ditunjukkan oleh setup_multiple_upload()

def analyze_multiple_charts(all_charts, analysis_mode):
    """Analyze multiple charts dengan 100% SOP"""
    batch_results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, chart_data in enumerate(all_charts):
        status_text.text(f"Analyzing {i+1}/{len(all_charts)}: {chart_data['name']}")
        
        # Analyze setiap chart dengan FASA 3 AI
        analysis_result = st.session_state.fasa3.analyze_uploaded_chart(chart_data['data'])
        
        if analysis_result:
            batch_results.append({
                'chart_name': chart_data['name'],
                'analysis': analysis_result,
                'image': chart_data['image']
            })
        
        progress_bar.progress((i + 1) / len(all_charts))
    
    status_text.text("✅ ANALYSIS COMPLETE!")
    return batch_results

def display_batch_results(batch_results):
    """Display results untuk multiple charts"""
    
    st.subheader("📈 BATCH ANALYSIS RESULTS")
    
    # Summary statistics
    total_charts = len(batch_results)
    strong_buy_signals = sum(1 for r in batch_results if any(sig['signal'] == 'STRONG_BUY' for sig in r['analysis']['buy_signals']))
    avg_confidence = np.mean([r['analysis']['confidence_score'] for r in batch_results])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Charts", total_charts)
    with col2:
        st.metric("Strong Buy Signals", strong_buy_signals)
    with col3:
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    # Detailed results untuk setiap chart
    for result in batch_results:
        with st.expander(f"📊 {result['chart_name']} - Confidence: {result['analysis']['confidence_score']:.1%}", expanded=False):
            
            col_img, col_analysis = st.columns(2)
            
            with col_img:
                img_rgb = cv2.cvtColor(result['image'], cv2.COLOR_BGR2RGB)
                st.image(img_rgb, use_column_width=True)
            
            with col_analysis:
                # Show key signals only
                analysis = result['analysis']
                
                # Trend
                if analysis['trend']:
                    trend = analysis['trend'][0]
                    st.write(f"**Trend:** {trend['direction']} ({trend['strength']}%)")
                
                # Buy Signals
                if analysis['buy_signals']:
                    best_signal = max(analysis['buy_signals'], key=lambda x: x['confidence'])
                    st.write(f"**Signal:** {best_signal['signal']} ({best_signal['confidence']:.1%})")
                
                # Entry Points
                if analysis['entry_points']:
                    best_entry = max(analysis['entry_points'], key=lambda x: x['confidence'])
                    st.write(f"**Best Entry:** {best_entry['level']} ({best_entry['confidence']:.1%})")
                
                # Risk
                st.write(f"**Risk:** {analysis['risk_assessment']['level']}")

# =========================================================================
# === UPDATE MAIN APP ===
# =========================================================================

def main():
    st.set_page_config(
        page_title="AI Trading System - ENHANCED UPLOAD",
        page_icon="🚀", 
        layout="wide"
    )
    
    st.title("🤖 AI TRADING SYSTEM - MULTIPLE UPLOAD ENABLED")
    st.markdown("---")
    
    # Navigation
    st.sidebar.title("NAVIGATION")
    page = st.sidebar.radio("PILIH FASA:", 
                           ["FASA 1 - MULTIPLE UPLOAD", 
                            "FASA 2 - INTEGRATED AI", 
                            "FASA 3 - BATCH ANALYSIS"])
    
    if page == "FASA 1 - MULTIPLE UPLOAD":
        render_fasa1_enhanced()
    elif page == "FASA 2 - INTEGRATED AI":
        render_fasa2()
    elif page == "FASA 3 - BATCH ANALYSIS":
        render_fasa3_enhanced()

def process_single_image_ai(img_data):
    """Process single image dan return learned rules"""
    # Implementation dari FASA 1 original
    rules_learned = []
    
    try:
        # Your existing FASA 1 image processing logic here
        img = img_data['image']
        
        # Simulate AI analysis
        rules_learned.append({
            'rule': f'PATTERN_FROM_{img_data["name"]}',
            'condition': 'image_analysis',
            'action': 'TRADING_STRATEGY', 
            'confidence': 0.75 + (random.random() * 0.2),
            'learned_by': 'enhanced_ai',
            'session_type': 'batch_processing'
        })
        
    except Exception as e:
        st.error(f"Error processing {img_data['name']}: {e}")
    
    return rules_learned

if __name__ == "__main__":
    main()
