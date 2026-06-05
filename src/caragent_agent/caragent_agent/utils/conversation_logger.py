"""对话日志记录模块 - Simplified"""
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable
import logging

class ConversationLogger:
    """对话日志记录器 - 只保留 foreground, background, physical"""
    
    def __init__(self, log_dir: str = "logs", session_name: Optional[str] = None, time_provider: Optional[Callable[[], str]] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log time provider (returns string representation of sim time)
        self.time_provider = time_provider
        
        # 生成会话ID和名称
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = session_name or f"session_{self.session_id}"
        
        # 创建本次会话的目录
        self.session_dir = self.log_dir / self.session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def get_session_dir(self) -> str:
        """获取会话目录路径"""
        return str(self.session_dir)
    
    def _append_to_log(self, filename: str, message: str):
        """Append message with timestamp to a log file in session directory"""
        # Print to stdout
        # print(message)
        
        log_file = self.session_dir / filename
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        sim_time_str = ""
        if self.time_provider:
            try:
                sim_time_str = f" [SimTime: {self.time_provider()}]"
            except Exception:
                pass
                
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}]{sim_time_str} {message}\n")
        except Exception as e:
            print(f"Failed to write to log file {filename}: {e}")

    def set_time_provider(self, provider: Callable[[], str]):
        """Set a callback to provide simulation time string"""
        self.time_provider = provider

    def log_foreground(self, message: str):
        """Log to foreground_workflow_agents.log"""
        self._append_to_log("foreground_workflow_agents.log", message)

    def log_background(self, message: str):
        """Log to background_agents.log"""
        self._append_to_log("background_agents.log", message)
        
    def log_physical(self, message: str):
        """Log to physical_layer.log"""
        self._append_to_log("physical_layer.log", message)

    def log_error(self, error_msg: str, error_type: str = "error", traceback: str = ""):
        """Log error"""
        content = f"ERROR TYPE: {error_type}\nMessage: {error_msg}\nTraceback:\n{traceback}"
        self._append_to_log("foreground_workflow_agents.log", content)
    
    def setup_library_logging(self):
        """Setup logging for external libraries to write to our log files"""
        log_file = self.session_dir / "foreground_workflow_agents.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('[%(asctime)s] [Lib] %(name)s %(levelname)s: %(message)s'))
        
        # Add to root logger to capture everything including dashscope
        logging.getLogger().addHandler(file_handler)
