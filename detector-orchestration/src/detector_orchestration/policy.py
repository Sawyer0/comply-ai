from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from pydantic import BaseModel, Field

from .config import Settings
from .models import ContentType


class CoverageMethod(str, Enum):
    REQUIRED_SET = "required_set"
    WEIGHTED = "weighted"
    TAXONOMY = "taxonomy"


class TenantPolicy(BaseModel):
    tenant_id: str
    bundle: str
    version: str = "v1"
    required_detectors: List[str] = Field(default_factory=list)
    optional_detectors: List[str] = Field(default_factory=list)
    coverage_method: CoverageMethod = CoverageMethod.REQUIRED_SET
    required_coverage: float = 1.0
    detector_weights: Dict[str, float] = Field(default_factory=dict)
    required_taxonomy_categories: List[str] = Field(default_factory=list)
    allowed_content_types: List[ContentType] = Field(default_factory=lambda: [ContentType.TEXT])


class PolicyDecision(BaseModel):
    selected_detectors: List[str]
    coverage_method: CoverageMethod
    coverage_requirements: Dict[str, float]
    routing_reason: str


class PolicyStore:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)

    def _path(self, tenant_id: str, bundle: str) -> Path:
        return self.base_dir / tenant_id / f"{bundle}.json"

    def get_policy(self, tenant_id: str, bundle: str) -> Optional[TenantPolicy]:
        p = self._path(tenant_id, bundle)
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        return TenantPolicy(**data)

    def save_policy(self, policy: TenantPolicy) -> None:
        p = self._path(policy.tenant_id, policy.bundle)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(policy.model_dump_json(indent=2), encoding="utf-8")

    def list_policies(self, tenant_id: str) -> List[str]:
        d = self.base_dir / tenant_id
        if not d.exists():
            return []
        return [fp.stem for fp in d.glob("*.json")]

    def delete_policy(self, tenant_id: str, bundle: str) -> bool:
        p = self._path(tenant_id, bundle)
        if p.exists():
            p.unlink()
            return True
        return False


class OPAPolicyEngine:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def evaluate(self, tenant_id: str, bundle: str, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.settings.config.opa_enabled or not self.settings.config.opa_url:
            return None
        url = f"{self.settings.config.opa_url.rstrip('/')}/v1/data/{tenant_id}/{bundle}/select"
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.post(url, json={"input": input_data})
                if resp.status_code == 200:
                    return resp.json().get("result")
        except Exception:
            return None
        return None


class PolicyManager:
    def __init__(self, store: PolicyStore, engine: OPAPolicyEngine):
        self.store = store
        self.engine = engine

    async def decide(self, tenant_id: str, bundle: str, content_type: ContentType, candidate_detectors: List[str]) -> PolicyDecision:
        # Query OPA first if configured
        opa_result = await self.engine.evaluate(
            tenant_id,
            bundle,
            {
                "content_type": content_type.value,
                "candidates": candidate_detectors,
            },
        )
        if opa_result and isinstance(opa_result, dict) and opa_result.get("selected"):
            selected = [d for d in opa_result.get("selected", []) if d in candidate_detectors]
            cov_method = CoverageMethod(opa_result.get("coverage_method", CoverageMethod.REQUIRED_SET))
            cov_req = opa_result.get("coverage_requirements", {"min_success_fraction": 1.0})
            return PolicyDecision(
                selected_detectors=selected,
                coverage_method=cov_method,
                coverage_requirements=cov_req,
                routing_reason="opa",
            )

        # Fallback: Tenant policy
        pol = self.store.get_policy(tenant_id, bundle)
        if pol:
            selected = [d for d in pol.required_detectors if d in candidate_detectors]
            return PolicyDecision(
                selected_detectors=selected,
                coverage_method=pol.coverage_method,
                coverage_requirements={
                    "min_success_fraction": pol.required_coverage,
                    "weights": pol.detector_weights,
                    "required_taxonomy_categories": pol.required_taxonomy_categories,
                },
                routing_reason="tenant_policy",
            )

        # Default: pass through all candidates, required set
        return PolicyDecision(
            selected_detectors=candidate_detectors,
            coverage_method=CoverageMethod.REQUIRED_SET,
            coverage_requirements={"min_success_fraction": 1.0},
            routing_reason="default",
        )


class PolicyValidationCLI:
    @staticmethod
    def validate_policy_file(path: str) -> Tuple[bool, str]:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            TenantPolicy(**data)
            return True, "ok"
        except Exception as e:  # noqa: BLE001
            return False, str(e)
