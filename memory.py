# =========================================================================
# === MEMORY SYSTEM YANG PERSISTENT ===
# =========================================================================

class SuperiorMemory:
    def __init__(self, base_dir="./ai_memory"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # FILE MEMORY - INI YANG BUAT AI TIDAK BODOH
        self.memory_file = self.base_dir / "ai_superior_memory.pkl"
        self.backup_file = self.base_dir / "ai_memory_backup.pkl"
        
        # LOAD MEMORY SETIAP KALI AI Dijalankan
        self.learned_knowledge, self.performance_stats = self.load_memory_verified()
        
        # TUNJUKKAN STATISTIK LEARNING
        st.sidebar.markdown("---")
        st.sidebar.subheader("🧠 AI LEARNING STATUS")
        st.sidebar.write(f"Rules: {len(self.learned_knowledge['sop_rules'])}")
        st.sidebar.write(f"Sessions: {self.performance_stats['learning_sessions']}")
        st.sidebar.write(f"Intelligence: {self.performance_stats['ai_intelligence']}%")

    def load_memory_verified(self):
        """Load memory dari file - INI YANG BUAT AI INGAT"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'rb') as f:
                    data = pickle.load(f)
                    st.success("✅ AI Memory Loaded - AI tidak lupa!")
                    return data, data['performance_stats']
            
            # Jika tiada memory, buat baru
            st.info("🆕 AI Memory Baru Dibuat")
            return self.create_new_memory()
            
        except Exception as e:
            st.error(f"❌ Memory load error: {e}")
            return self.create_new_memory()

    def save_memory_guaranteed(self):
        """Simpan memory ke file - INI YANG BUAT AI TERUS BELAJAR"""
        try:
            # Update stats
            self.learned_knowledge['performance_stats'] = self.performance_stats
            self.learned_knowledge['last_updated'] = time.time()
            
            # Simpan ke file
            with open(self.memory_file, 'wb') as f:
                pickle.dump(self.learned_knowledge, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Backup
            with open(self.backup_file, 'wb') as f:
                pickle.dump(self.learned_knowledge, f, protocol=pickle.HIGHEST_PROTOCOL)
                
            return True
            
        except Exception as e:
            st.error(f"❌ Memory save failed: {e}")
            return False
