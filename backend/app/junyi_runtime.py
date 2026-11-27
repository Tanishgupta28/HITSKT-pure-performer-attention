"""Server configuration and availability for the single-learner Junyi prototype."""

import logging
from dataclasses import dataclass, field

from app.config import ROOT
from app.junyi_bank import STAGES, JunyiBank, JunyiMappings, load_bank, load_mappings
from app.junyi_model import OriginalHiTSKT, SkillEstimator, read_baseline
from app.junyi_paper import BUCKETS
from app.junyi_resources import Gemini, StudyRecommender
from app.junyi_study_policy import FOREST_POLICY, PaperRecommender


logger = logging.getLogger(__name__)


@dataclass
class JunyiRuntime:
    bank: JunyiBank | None
    mappings: JunyiMappings | None
    estimator: SkillEstimator | None
    gemini: Gemini
    recommender: StudyRecommender
    quotas: dict[str, int]
    issues: list[str] = field(default_factory=list)
    study_policy: PaperRecommender | None = None
    study_policy_issue: str | None = None

    @property
    def ready(self):
        return self.bank is not None and self.mappings is not None and self.estimator is not None

    @classmethod
    def from_settings(cls, settings):
        key = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else None
        gemini = Gemini(key, settings.gemini_model)
        runtime = cls(None, None, None, gemini, StudyRecommender(gemini), dict(zip(BUCKETS, settings.junyi_quotas, strict=True)))
        if not settings.junyi_student_account_id:
            return runtime
        if (settings.junyi_stage not in (*STAGES, None) or not 0 <= settings.junyi_weak_threshold < settings.junyi_strong_threshold <= 1
                or min(settings.junyi_quotas) < 0 or sum(settings.junyi_quotas) != 20):
            runtime.issues.append("Junyi stage, thresholds or 20-question quotas need correction.")
            return runtime
        try:
            runtime.mappings = load_mappings(ROOT / "reports" / "datasets" / "junyi", settings.junyi_ranking_file)
            runtime.bank = load_bank(settings.junyi_bank_file, runtime.mappings)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            runtime.issues.append("The approved Junyi bank or mappings are missing." if isinstance(exc, OSError)
                                  else "The Junyi bank or rankings did not pass verification.")
        if not settings.junyi_history_file.is_file() or not settings.junyi_history_sha256:
            runtime.issues.append("The authorized Student 1 history and its verified hash are required.")
        if not settings.junyi_checkpoint.is_file():
            runtime.issues.append("The original Junyi HiTSKT checkpoint is required.")
        if runtime.issues:
            return runtime
        try:
            baseline = read_baseline(settings.junyi_history_file, settings.junyi_history_sha256, runtime.mappings)
            predictor = OriginalHiTSKT(settings.junyi_checkpoint)
            predictor.load()
            runtime.estimator = SkillEstimator(baseline, runtime.bank, runtime.mappings, predictor,
                                               settings.junyi_weak_threshold, settings.junyi_strong_threshold)
        except ImportError:
            runtime.issues.append("Install the model-serving dependencies to enable Junyi inference.")
        except (OSError, ValueError, KeyError, TypeError, RuntimeError):
            runtime.issues.append("The supplied Junyi checkpoint or approved history did not pass verification.")
        if runtime.ready and settings.junyi_study_forest_file:
            try:
                runtime.study_policy = PaperRecommender.load(
                    settings.junyi_study_forest_file, settings.junyi_ranking_file, runtime.mappings,
                )
                logger.info("Loaded Junyi study policy %s", FOREST_POLICY)
            except (ImportError, OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError) as exc:
                runtime.study_policy_issue = "study_forest_load_failed"
                logger.warning("Junyi study forest unavailable (%s). Using the original checklist.", type(exc).__name__)
        return runtime
