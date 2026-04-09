"""
Structured Logging pour SCAF-LS
Support JSON, multiples niveaux, contexte enrichi
"""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path
import traceback


class StructuredLogger:
    """Logger structuré avec support JSON et contexte enrichi"""
    
    def __init__(self, name: str, log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(name)
        self.context: Dict[str, Any] = {}
        
        # Handlers
        self._setup_handlers()
        
    def _setup_handlers(self):
        """Configure les handlers JSON et console"""
        self.logger.setLevel(logging.DEBUG)
        
        # File handler - JSON
        json_handler = logging.FileHandler(
            self.log_dir / f"{self.name}_{datetime.now():%Y%m%d_%H%M}.json"
        )
        json_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(json_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s | [%(levelname)s] %(name)s: %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
    def set_context(self, **kwargs):
        """Ajouter du contexte persistant aux logs"""
        self.context.update(kwargs)
        
    def clear_context(self):
        """Effacer le contexte"""
        self.context.clear()
        
    def _log(self, level: int, message: str, extra: Optional[Dict] = None, 
             exc_info: bool = False):
        """Log interne avec support du contexte"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': logging.getLevelName(level),
            'logger': self.name,
            'message': message,
            **self.context,
        }
        
        if extra:
            log_data['extra'] = extra
            
        if exc_info:
            log_data['traceback'] = traceback.format_exc()
        
        # Créer un record de log personnalisé
        record = logging.LogRecord(
            name=self.name,
            level=level,
            pathname="",
            lineno=0,
            msg=json.dumps(log_data),
            args=(),
            exc_info=None
        )
        record.log_data = log_data
        
        self.logger.handle(record)
        
    def debug(self, message: str, **kwargs):
        """Log debug"""
        self._log(logging.DEBUG, message, extra=kwargs)
        
    def info(self, message: str, **kwargs):
        """Log info"""
        self._log(logging.INFO, message, extra=kwargs)
        
    def warning(self, message: str, **kwargs):
        """Log warning"""
        self._log(logging.WARNING, message, extra=kwargs)
        
    def error(self, message: str, exc_info: bool = False, **kwargs):
        """Log error"""
        self._log(logging.ERROR, message, extra=kwargs, exc_info=exc_info)
        
    def critical(self, message: str, exc_info: bool = False, **kwargs):
        """Log critique"""
        self._log(logging.CRITICAL, message, extra=kwargs, exc_info=exc_info)
        

class JSONFormatter(logging.Formatter):
    """Formateur JSON pour les logs"""
    
    def format(self, record: logging.LogRecord) -> str:
        if hasattr(record, 'log_data'):
            return json.dumps(record.log_data)
        
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


# Instance globale
_logger_instance: Optional[StructuredLogger] = None

def get_logger(name: str = "scaf-ls") -> StructuredLogger:
    """Récupérer l'instance globale du logger"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = StructuredLogger(name)
    return _logger_instance
