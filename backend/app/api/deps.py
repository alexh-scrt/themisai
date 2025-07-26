# app/api/deps.py
async def get_case_service() -> CaseService:
    return CaseService(
        case_repository=get_case_repository(),
        embedding_service=get_embedding_service()
    )