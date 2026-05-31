"""
Background scheduler for periodic JARVIS tasks.
"""
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from backend.logger import get_logger
from configs.settings import get_settings

logger = get_logger(__name__)


class JarvisScheduler:
    """Manages background tasks for Jarvis."""

    def __init__(self, db, vector_store, memory_extractor, analytics_manager, settings=None):
        self._db = db
        self._vector_store = vector_store
        self._memory_extractor = memory_extractor
        self._analytics_manager = analytics_manager
        self._settings = settings or get_settings()

        # Phase 3 Enhancements
        self._analytics_engine = None
        self._proactive_layer = None
        
        # Load enhanced components if available
        try:
            from backend.analytics_engine import AnalyticsEngine
            from backend.proactive_layer import ProactiveLayer
            self._analytics_engine = AnalyticsEngine(self._db, self._analytics_manager, self._settings)
            self._proactive_layer = ProactiveLayer(self._db, self._analytics_manager, self._settings)
        except ImportError:
            logger.warning("Phase 3 components (Analytics/Proactive) not found. Scheduler will run in legacy mode.")

        self._scheduler = BackgroundScheduler()
        self._setup_jobs()

    def _setup_jobs(self) -> None:
        """Configure scheduled jobs based on settings."""
        if not self._settings.scheduler.enabled:
            logger.info("Scheduler disabled in settings.")
            return

        # 1. Database Backups (runs every 12h by default)
        self._scheduler.add_job(
            self._backup_database,
            'interval',
            hours=self._settings.scheduler.backup_interval_hours,
            id='db_backup',
            replace_existing=True
        )

        # 2. Daily Goal & Habit Check (Midnight)
        self._scheduler.add_job(
            self._check_daily_status,
            'cron',
            hour=0,
            minute=5,
            id='daily_status_check',
            replace_existing=True
        )
        
        # 3. Memory Maintenance (Decay & Consolidation) (Weekly)
        # Run at 3 AM on Sunday
        self._scheduler.add_job(
            self._run_memory_maintenance,
            'cron',
            day_of_week='sun',
            hour=3,
            minute=0,
            id='memory_maintenance',
            replace_existing=True
        )
        
        # 4. Weekly Analytics Report (Sunday 8 AM)
        if self._analytics_engine:
            self._scheduler.add_job(
                self._generate_weekly_report,
                'cron',
                day_of_week='sun',
                hour=8,
                minute=0,
                id='weekly_analytics_report',
                replace_existing=True
            )
            
        # Parse times for daily proactive tasks
        try:
            m_hour, m_minute = map(int, self._settings.scheduler.morning_summary_time.split(":"))
            n_hour, n_minute = map(int, self._settings.scheduler.nightly_reflection_time.split(":"))
            
            # 5. Morning Briefing
            if self._proactive_layer and getattr(self._settings.proactive, 'enable_morning_briefing', True):
                self._scheduler.add_job(
                    self._generate_morning_briefing,
                    trigger=CronTrigger(hour=m_hour, minute=m_minute),
                    id='morning_briefing',
                    name="Morning Briefing",
                    replace_existing=True
                )
                
            # 6. Evening Nudge
            if self._proactive_layer and getattr(self._settings.proactive, 'enable_evening_nudge', True):
                self._scheduler.add_job(
                    self._generate_evening_nudge,
                    trigger=CronTrigger(hour=n_hour, minute=n_minute),
                    id='evening_nudge',
                    name="Evening Nudge",
                    replace_existing=True
                )
        except Exception as e:
            logger.error(f"Failed to parse scheduler times: {e}")

    def start(self):
        """Start the background scheduler."""
        if self._settings.scheduler.enabled:
            logger.info("Starting Jarvis Scheduler...")
            self._scheduler.start()
        else:
            logger.info("Scheduler is disabled, not starting.")

    def stop(self):
        """Stop the background scheduler."""
        if self._scheduler.running:
            logger.info("Stopping Jarvis Scheduler...")
            self._scheduler.shutdown()

    # -------------------------------------------------------------------
    # Job Functions
    # -------------------------------------------------------------------

    def _backup_database(self):
        logger.info("Running scheduled database backup...")
        self._db.backup()

    def _check_daily_status(self):
        logger.info("Running daily status check...")
        
        # Mark overdue goals
        self._db.execute_write(
            "UPDATE goals SET status = 'failed' "
            "WHERE status = 'active' AND deadline < date('now')"
        )
        
        # Check habit streaks
        self._analytics_manager.update_habit_streaks()

    def _run_memory_maintenance(self):
        """Phase 3: Run decay rules and consolidation."""
        logger.info("Running memory maintenance...")
        try:
            # Apply decay
            decay_results = self._memory_extractor.apply_decay_rules(self._db)
            
            # Consolidate
            consolidation_results = self._memory_extractor.consolidate_memories(
                self._vector_store, self._db
            )
            
            # Clean up old inbox messages
            if self._proactive_layer:
                cleared = self._proactive_layer.clear_old_inbox()
                logger.info("Cleared %d old inbox messages", cleared)
                
        except Exception as e:
            logger.error("Memory maintenance failed: %s", e)

    def _generate_weekly_report(self):
        """Phase 3: Generate the weekly analytics report."""
        logger.info("Running weekly analytics report...")
        try:
            if self._analytics_engine:
                report = self._analytics_engine.generate_weekly_report()
                # Create a high priority inbox item for it
                if self._proactive_layer and report:
                    from backend.proactive_layer import InboxItem
                    import uuid
                    from datetime import datetime, timezone
                    
                    item = InboxItem(
                        id=str(uuid.uuid4()),
                        type="weekly_report",
                        title=f"Weekly Review ({report.week_start} to {report.week_end})",
                        content=f"Your weekly report is ready.\nInsight: {report.behavioral_insight}",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        priority="high"
                    )
                    self._proactive_layer._store_inbox_item(item)
        except Exception as e:
            logger.error("Weekly report generation failed: %s", e)
            
    def _generate_morning_briefing(self):
        """Phase 3: Generate daily morning briefing."""
        logger.info("Generating morning briefing...")
        try:
            if self._proactive_layer:
                self._proactive_layer.morning_briefing()
        except Exception as e:
            logger.error("Morning briefing generation failed: %s", e)
            
    def _generate_evening_nudge(self):
        """Phase 3: Generate evening nudges."""
        logger.info("Generating evening nudges...")
        try:
            if self._proactive_layer:
                self._proactive_layer.evening_nudge()
        except Exception as e:
            logger.error("Evening nudge generation failed: %s", e)
