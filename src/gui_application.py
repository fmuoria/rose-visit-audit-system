"""
ROSE Women Leaders Visit Audit System - GUI Application
Tkinter-based interface for easy CSV upload and audit result generation
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import os
from datetime import datetime
import threading
from audit_system import VisitAuditSystem  # Import our audit system

class ROSEAuditGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ROSE Visit Audit System")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.input_file = tk.StringVar()
        self.output_folder = tk.StringVar()
        self.similarity_threshold = tk.DoubleVar(value=0.7)
        self.delay_threshold = tk.IntVar(value=7)
        self.auditor = None
        
        # Set default output folder to user's Downloads
        default_output = os.path.join(os.path.expanduser("~"), "Downloads")
        self.output_folder.set(default_output)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface"""
        
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill='x', padx=0, pady=0)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="🌹 ROSE Women Leaders Visit Audit System", 
            font=("Arial", 18, "bold"),
            fg='white', 
            bg='#2c3e50'
        )
        title_label.pack(expand=True)
        
        # Main content frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # File Selection Section
        file_section = tk.LabelFrame(
            main_frame, 
            text="📁 File Selection", 
            font=("Arial", 12, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        file_section.pack(fill='x', pady=(0, 15))
        
        # Input file selection
        tk.Label(file_section, text="Select Visit Data CSV:", bg='#f0f0f0').pack(anchor='w', padx=10, pady=(10, 5))
        
        input_frame = tk.Frame(file_section, bg='#f0f0f0')
        input_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        input_entry = tk.Entry(input_frame, textvariable=self.input_file, width=60, state='readonly')
        input_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))
        
        input_button = tk.Button(
            input_frame, 
            text="Browse", 
            command=self.browse_input_file,
            bg='#3498db',
            fg='white',
            font=("Arial", 10, "bold"),
            width=10
        )
        input_button.pack(side='right')
        
        # Output folder selection
        tk.Label(file_section, text="Output Folder:", bg='#f0f0f0').pack(anchor='w', padx=10, pady=(5, 5))
        
        output_frame = tk.Frame(file_section, bg='#f0f0f0')
        output_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        output_entry = tk.Entry(output_frame, textvariable=self.output_folder, width=60, state='readonly')
        output_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))
        
        output_button = tk.Button(
            output_frame, 
            text="Browse", 
            command=self.browse_output_folder,
            bg='#3498db',
            fg='white',
            font=("Arial", 10, "bold"),
            width=10
        )
        output_button.pack(side='right')
        
        # Configuration Section
        config_section = tk.LabelFrame(
            main_frame, 
            text="⚙️ Audit Configuration", 
            font=("Arial", 12, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        config_section.pack(fill='x', pady=(0, 15))
        
        # Similarity threshold
        similarity_frame = tk.Frame(config_section, bg='#f0f0f0')
        similarity_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(
            similarity_frame, 
            text="Similarity Threshold (0.1 - 1.0):", 
            bg='#f0f0f0',
            width=25,
            anchor='w'
        ).pack(side='left')
        
        similarity_scale = tk.Scale(
            similarity_frame,
            from_=0.1,
            to=1.0,
            resolution=0.1,
            orient='horizontal',
            variable=self.similarity_threshold,
            bg='#f0f0f0',
            length=200
        )
        similarity_scale.pack(side='left', padx=10)
        
        tk.Label(
            similarity_frame, 
            text="(70% = 0.7 recommended)", 
            bg='#f0f0f0',
            fg='#7f8c8d',
            font=("Arial", 9)
        ).pack(side='left', padx=10)
        
        # Upload delay threshold
        delay_frame = tk.Frame(config_section, bg='#f0f0f0')
        delay_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        tk.Label(
            delay_frame, 
            text="Upload Delay Threshold (days):", 
            bg='#f0f0f0',
            width=25,
            anchor='w'
        ).pack(side='left')
        
        delay_spinbox = tk.Spinbox(
            delay_frame,
            from_=1,
            to=30,
            textvariable=self.delay_threshold,
            width=10,
            font=("Arial", 10)
        )
        delay_spinbox.pack(side='left', padx=10)
        
        tk.Label(
            delay_frame, 
            text="(7 days recommended)", 
            bg='#f0f0f0',
            fg='#7f8c8d',
            font=("Arial", 9)
        ).pack(side='left', padx=10)
        
        # Action Section
        action_section = tk.LabelFrame(
            main_frame, 
            text="🚀 Run Audit", 
            font=("Arial", 12, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        action_section.pack(fill='x', pady=(0, 15))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            action_section,
            variable=self.progress_var,
            maximum=100,
            length=400,
            mode='determinate'
        )
        self.progress_bar.pack(pady=10)
        
        # Status label
        self.status_label = tk.Label(
            action_section,
            text="Ready to run audit",
            bg='#f0f0f0',
            fg='#2c3e50',
            font=("Arial", 10)
        )
        self.status_label.pack(pady=(0, 10))
        
        # Run audit button
        self.run_button = tk.Button(
            action_section,
            text="🔍 Run Audit Analysis",
            command=self.run_audit,
            bg='#27ae60',
            fg='white',
            font=("Arial", 14, "bold"),
            height=2,
            width=20,
            cursor='hand2'
        )
        self.run_button.pack(pady=10)
        
        # Results Section
        results_section = tk.LabelFrame(
            main_frame, 
            text="📊 Audit Results Preview", 
            font=("Arial", 12, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        results_section.pack(fill='both', expand=True)
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(
            results_section,
            wrap=tk.WORD,
            width=80,
            height=15,
            font=("Consolas", 9),
            bg='#2c3e50',
            fg='#ecf0f1',
            insertbackground='white'
        )
        self.results_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Initially disable the text area
        self.results_text.config(state='disabled')
        
        # Footer
        footer_frame = tk.Frame(self.root, bg='#34495e', height=30)
        footer_frame.pack(fill='x', side='bottom')
        footer_frame.pack_propagate(False)
        
        footer_label = tk.Label(
            footer_frame,
            text="ROSE Women Leaders Program - Visit Audit System v1.0",
            bg='#34495e',
            fg='#bdc3c7',
            font=("Arial", 9)
        )
        footer_label.pack(expand=True)
    
    def browse_input_file(self):
        """Browse and select input CSV file"""
        file_types = [
            ("CSV files", "*.csv"),
            ("Excel files", "*.xlsx"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Visit Data File",
            filetypes=file_types
        )
        
        if filename:
            self.input_file.set(filename)
            self.update_results_preview("📁 Input file selected: " + os.path.basename(filename))
    
    def browse_output_folder(self):
        """Browse and select output folder"""
        folder = filedialog.askdirectory(
            title="Select Output Folder"
        )
        
        if folder:
            self.output_folder.set(folder)
            self.update_results_preview("📁 Output folder selected: " + folder)
    
    def update_results_preview(self, message):
        """Update results preview with a message"""
        self.results_text.config(state='normal')
        self.results_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
        self.results_text.see(tk.END)
        self.results_text.config(state='disabled')
        self.root.update()
    
    def update_progress(self, value, status):
        """Update progress bar and status"""
        self.progress_var.set(value)
        self.status_label.config(text=status)
        self.root.update()
    
    def validate_inputs(self):
        """Validate user inputs before running audit"""
        if not self.input_file.get():
            messagebox.showerror("Error", "Please select an input CSV file!")
            return False
        
        if not os.path.exists(self.input_file.get()):
            messagebox.showerror("Error", "Input file does not exist!")
            return False
        
        if not self.output_folder.get():
            messagebox.showerror("Error", "Please select an output folder!")
            return False
        
        if not os.path.exists(self.output_folder.get()):
            messagebox.showerror("Error", "Output folder does not exist!")
            return False
        
        return True
    
    def run_audit_thread(self):
        """Run audit in separate thread to prevent UI freezing"""
        try:
            # Initialize audit system
            self.update_progress(10, "Initializing audit system...")
            self.auditor = VisitAuditSystem(
                similarity_threshold=self.similarity_threshold.get(),
                upload_delay_threshold=self.delay_threshold.get()
            )
            
            # Load data
            self.update_progress(20, "Loading and preparing data...")
            self.update_results_preview("📊 Loading data from: " + os.path.basename(self.input_file.get()))
            
            self.auditor.load_data(self.input_file.get())
            total_visits = len(self.auditor.df)
            total_trainers = self.auditor.df['Visitation: Created By'].nunique()
            
            self.update_results_preview(f"✅ Data loaded successfully: {total_visits} visits, {total_trainers} trainers")
            
            # Run different audit checks
            self.update_progress(30, "Analyzing location patterns...")
            self.update_results_preview("🗺️  Running location similarity analysis...")
            location_flags = self.auditor.audit_location_similarity()
            
            self.update_progress(45, "Analyzing story authenticity...")
            self.update_results_preview("📝 Running story similarity analysis...")
            story_flags = self.auditor.audit_story_similarity()
            
            self.update_progress(60, "Analyzing upload delays...")
            self.update_results_preview("⏱️  Running upload delay analysis...")
            delay_flags = self.auditor.audit_upload_delays()
            
            self.update_progress(70, "Analyzing income patterns...")
            self.update_results_preview("💰 Running income growth analysis...")
            income_flags = self.auditor.audit_income_growth_anomalies()
            
            self.update_progress(80, "Analyzing temporal patterns...")
            self.update_results_preview("⏰ Running temporal clustering analysis...")
            temporal_flags = self.auditor.audit_temporal_clustering()
            
            self.update_progress(90, "Calculating confidence scores...")
            self.update_results_preview("🎯 Calculating visit and trainer confidence scores...")
            visit_scores = self.auditor.calculate_visit_confidence()
            trainer_scores = self.auditor.calculate_trainer_confidence()
            
            # Generate output files
            self.update_progress(95, "Generating output files...")
            self.update_results_preview("💾 Generating CSV output files...")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create comprehensive results
            results_summary = {
                'location_flags': len(location_flags),
                'story_flags': len(story_flags),
                'delay_flags': len(delay_flags),
                'income_flags': len(income_flags),
                'temporal_flags': len(temporal_flags),
                'total_visits': total_visits,
                'total_trainers': total_trainers,
                'low_confidence_visits': len([v for v in visit_scores if v['confidence_score'] < 60]),
                'high_priority_trainers': len([t for t in trainer_scores if t['audit_priority'] == 'HIGH'])
            }
            
            # Save results to CSV files
            output_files = []
            
            # 1. Visit Scores CSV
            visit_df = pd.DataFrame(visit_scores)
            visit_file = os.path.join(self.output_folder.get(), f"visit_confidence_scores_{timestamp}.csv")
            visit_df.to_csv(visit_file, index=False)
            output_files.append(visit_file)
            
            # 2. Trainer Scores CSV
            trainer_df = pd.DataFrame(trainer_scores)
            trainer_file = os.path.join(self.output_folder.get(), f"trainer_audit_priority_{timestamp}.csv")
            trainer_df.to_csv(trainer_file, index=False)
            output_files.append(trainer_file)
            
            # 3. Detailed Flags CSV
            flags_data = []
            
            # Add location flags
            for flag in location_flags:
                flags_data.append({
                    'Flag_Type': 'Location_Similarity',
                    'Trainer': flag['trainer'],
                    'Severity': 'HIGH' if flag['similarity_percentage'] > 0.5 else 'MEDIUM',
                    'Details': f"Similarity: {flag['similarity_percentage']:.1%}, Pairs: {flag['flagged_pairs']}"
                })
            
            # Add story flags
            for flag in story_flags:
                severity = 'HIGH' if flag['authenticity_score'] < 30 else 'MEDIUM' if flag['authenticity_score'] < 60 else 'LOW'
                flags_data.append({
                    'Flag_Type': 'Story_Similarity',
                    'Trainer': flag['trainer'],
                    'Severity': severity,
                    'Details': f"Authenticity: {flag['authenticity_score']:.1f}%, Similar pairs: {len(flag['similar_stories'])}"
                })
            
            # Add income flags
            for flag in income_flags:
                severity = 'HIGH' if flag['suspicious_score'] > 70 else 'MEDIUM' if flag['suspicious_score'] > 40 else 'LOW'
                flags_data.append({
                    'Flag_Type': 'Income_Anomaly',
                    'Trainer': flag['trainer'],
                    'Severity': severity,
                    'Details': f"Suspicion: {flag['suspicious_score']:.1f}%, Extreme growth: {flag['extreme_growth_count']}"
                })
            
            # Add temporal flags
            for flag in temporal_flags:
                severity = 'HIGH' if flag['temporal_suspicion_score'] > 70 else 'MEDIUM' if flag['temporal_suspicion_score'] > 40 else 'LOW'
                flags_data.append({
                    'Flag_Type': 'Temporal_Clustering',
                    'Trainer': flag['trainer'],
                    'Severity': severity,
                    'Details': f"Suspicion: {flag['temporal_suspicion_score']:.1f}%, Monday visits: {flag['monday_percentage']:.1f}%"
                })
            
            if flags_data:
                flags_df = pd.DataFrame(flags_data)
                flags_file = os.path.join(self.output_folder.get(), f"audit_flags_detailed_{timestamp}.csv")
                flags_df.to_csv(flags_file, index=False)
                output_files.append(flags_file)
            
            # 4. Summary Report CSV
            summary_df = pd.DataFrame([results_summary])
            summary_file = os.path.join(self.output_folder.get(), f"audit_summary_{timestamp}.csv")
            summary_df.to_csv(summary_file, index=False)
            output_files.append(summary_file)
            
            self.update_progress(100, "Audit completed successfully!")
            
            # Display final results
            self.update_results_preview("=" * 60)
            self.update_results_preview("🎉 AUDIT COMPLETED SUCCESSFULLY!")
            self.update_results_preview("=" * 60)
            self.update_results_preview(f"📊 Total visits analyzed: {total_visits}")
            self.update_results_preview(f"👥 Total trainers analyzed: {total_trainers}")
            self.update_results_preview(f"🔍 Location flags: {results_summary['location_flags']}")
            self.update_results_preview(f"📝 Story flags: {results_summary['story_flags']}")
            self.update_results_preview(f"💰 Income flags: {results_summary['income_flags']}")
            self.update_results_preview(f"⏰ Temporal flags: {results_summary['temporal_flags']}")
            self.update_results_preview(f"⚠️  Low confidence visits: {results_summary['low_confidence_visits']}")
            self.update_results_preview(f"🔴 High priority trainers: {results_summary['high_priority_trainers']}")
            self.update_results_preview("")
            self.update_results_preview("📁 OUTPUT FILES GENERATED:")
            for file in output_files:
                self.update_results_preview(f"   • {os.path.basename(file)}")
            
            # Show completion dialog
            self.root.after(0, lambda: messagebox.showinfo(
                "Audit Complete", 
                f"Audit completed successfully!\n\n"
                f"Generated {len(output_files)} CSV files in:\n{self.output_folder.get()}\n\n"
                f"Key findings:\n"
                f"• {results_summary['low_confidence_visits']} low confidence visits\n"
                f"• {results_summary['high_priority_trainers']} high priority trainers for audit"
            ))
            
        except Exception as e:
            error_msg = f"Error during audit: {str(e)}"
            self.update_results_preview(f"❌ {error_msg}")
            self.root.after(0, lambda: messagebox.showerror("Audit Error", error_msg))
        
        finally:
            # Re-enable the run button
            self.root.after(0, lambda: self.run_button.config(state='normal'))
    
    def run_audit(self):
        """Main function to run the audit"""
        if not self.validate_inputs():
            return
        
        # Clear previous results
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state='disabled')
        
        # Reset progress
        self.progress_var.set(0)
        self.status_label.config(text="Starting audit...")
        
        # Disable run button during processing
        self.run_button.config(state='disabled')
        
        # Run audit in separate thread
        audit_thread = threading.Thread(target=self.run_audit_thread)
        audit_thread.daemon = True
        audit_thread.start()

def main():
    """Main function to run the GUI application"""
    root = tk.Tk()
    
    # Set application icon (if available)
    try:
        # You can add an icon file here
        # root.iconbitmap('icon.ico')
        pass
    except:
        pass
    
    app = ROSEAuditGUI(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    # Start the GUI
    root.mainloop()

if __name__ == "__main__":
    main()
