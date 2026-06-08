import time
import pandas as pd

from config import NAVER_MAP_URL, OUTPUT_CSV
from crawler import create_driver, switch_to_entry_iframe, click_review_tab, collect_all_reviews

# 네이버 지도 리뷰 크롤러 메인 함수
# 이 파일을 실행해야 함
def main():
    driver = create_driver()

    try:
        print("[INFO] 페이지 접속 중...")
        driver.get(NAVER_MAP_URL)
        time.sleep(6)

        switch_to_entry_iframe(driver)
        click_review_tab(driver)

        data = collect_all_reviews(driver)
        print(f"\n[INFO] 수집 완료: {len(data)}개")

        df = pd.DataFrame(data)

        if not df.empty:
            df = (
                df.drop_duplicates(
                    subset=["계정 ID", "방문 날짜", "리뷰 내용"],
                    keep="first"
                )
                .sort_values(
                    by=["방문 날짜", "계정 ID"],
                    ascending=[False, True]
                )
            )

        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"[DONE] 저장 완료: {len(df)}개 → {OUTPUT_CSV}")
        print(df.head())

    finally:
        driver.quit()


if __name__ == "__main__":
    main()