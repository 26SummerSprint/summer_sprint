"""
POST /label, POST /label/batch.

GroundTruthLabeler(핵심 라벨링 로직)를 그대로 호출만 한다 - 이 라우터는
HTTP 입출력 변환만 담당한다.
"""

from fastapi import APIRouter, Depends, HTTPException

from ...labeler import GroundTruthLabeler, GroundTruthLabelerError
from ..deps import verify_api_key
from ..schemas import (
    BatchLabelItemOut,
    BatchLabelRequest,
    BatchLabelResponse,
    LabelRequest,
    LabelResultOut,
)

router = APIRouter(dependencies=[Depends(verify_api_key)])

# 프로세스당 1회만 생성 (내부에 동시 요청 제한용 세마포어를 갖고 있으므로 재사용).
_labeler = GroundTruthLabeler()


def _to_label_result_out(result, keep_raw: bool) -> LabelResultOut:
    return LabelResultOut(
        score=result.score,
        decision=result.decision,
        label=result.label,
        tag=result.tag,
        reason=result.reason,
        analysis=result.analysis if keep_raw else None,
        raw_response=result.raw_response if keep_raw else None,
        parse_warnings=result.parse_warnings,
    )


@router.post("/label", response_model=LabelResultOut)
async def label(req: LabelRequest) -> LabelResultOut:
    """프로필-논문 한 쌍을 라벨링한다."""
    try:
        result = await _labeler.label(req.profile, req.title, req.abstract)
    except GroundTruthLabelerError as e:
        print("=" * 60)
        print("GEMINI ERROR:")
        print(str(e))
        print("=" * 60)

        raise HTTPException(
            status_code=502,
            detail=str(e),
        ) from e

    except Exception as e:
        print("=" * 60)
        print("UNEXPECTED ERROR:")
        import traceback
        traceback.print_exc()
        print("=" * 60)

        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e
    return _to_label_result_out(result, keep_raw=True)


@router.post("/label/batch", response_model=BatchLabelResponse)
async def label_batch(req: BatchLabelRequest) -> BatchLabelResponse:
    """
    같은 프로필에 대해 여러 논문을 한 번에 라벨링한다.
    동시 요청 수는 GroundTruthLabeler 내부 설정(MAX_CONCURRENT_REQUESTS)을 따른다.
    개별 논문 라벨링이 실패해도 전체 요청은 502로 실패하지 않고, 해당 항목의
    "error" 필드에 이유가 담긴 채로 200이 반환된다.
    """
    if not req.papers:
        return BatchLabelResponse(
            count=0, success_count=0, failed_count=0, positive_count=0, results=[]
        )

    raw_results = await _labeler.label_batch(
        req.profile,
        req.papers,
        title_key=req.title_key,
        abstract_key=req.abstract_key,
        id_key=req.id_key,
    )

    items = []
    success_count = 0
    positive_count = 0
    for r in raw_results:
        if r["result"] is not None:
            success_count += 1
            out = _to_label_result_out(r["result"], keep_raw=req.keep_raw)
            if out.label == 1:
                positive_count += 1
        else:
            out = None
        items.append(
            BatchLabelItemOut(
                arxiv_id=r["arxiv_id"],
                title=r.get("title"),
                result=out,
                error=r["error"],
            )
        )

    return BatchLabelResponse(
        count=len(items),
        success_count=success_count,
        failed_count=len(items) - success_count,
        positive_count=positive_count,
        results=items,
    )
