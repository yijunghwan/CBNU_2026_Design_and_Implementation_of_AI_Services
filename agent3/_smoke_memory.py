"""멀티턴 메모리 플로우 검증 (임시)."""

from agent3.service import AgentService


def show(label, res):
    print("=" * 72)
    print(f"[{label}] memory={res['memory']} plan={res['plan']}")
    print("-" * 72)
    print(res["response"])


if __name__ == "__main__":
    svc = AgentService()
    code = "user-abc"

    show("1. 장기기억 켜기", svc.process(user_id="u1", query="장기기억 켜줘", user_code=code))
    show("2. 동의", svc.process(user_id="u1", query="동의", user_code=code))
    show("3. 프로필 변경", svc.process(user_id="u1", query="개인정보 변경: 만 27세, 서울 거주, 취업준비중, 미혼, 무주택", user_code=code))
    show("4. 동의", svc.process(user_id="u1", query="동의", user_code=code))
    show("5. 프로필 조회", svc.process(user_id="u1", query="내 개인정보 보여줘", user_code=code))
    show("6. 맞춤 추천", svc.process(user_id="u1", query="나한테 맞는 정책 추천해줘", user_code=code))
