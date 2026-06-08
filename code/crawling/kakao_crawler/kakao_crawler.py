import re
import time
from datetime import datetime
from typing import List, Optional, Sequence

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import (
    StaleElementReferenceException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


# =====================================
# 수집 대상 설정
# =====================================
URL = "https://place.map.kakao.com/10199904#review"

# 수집할 리뷰 날짜 범위 (MAX_REVIEWS=None일 때 이 날짜 이전 리뷰가 나오면 수집 중단)
START_DATE = datetime(2025, 1, 1)

# =====================================
# 크롤링 동작 설정
# =====================================
MAX_REVIEWS   = None  # 최대 수집 리뷰 수 (None이면 START_DATE 기준으로 수집)
MAX_ROUNDS    = 30    # 최대 라운드 수
NO_NEW_LIMIT  = 3     # 신규 리뷰 없을 때 허용 횟수 (초과 시 종료)

# =====================================
# 출력 파일 경로
# =====================================
OUTPUT_CSV = f"kakao_reviews_{int(time.time())}.csv"

# =====================================
# 내부 상수
# =====================================
DEFAULT_WAIT_SECONDS = 5
PAGE_LOAD_SLEEP = 3.0
CLICK_SLEEP = 1.0


# =====================================
# 드라이버 생성
# =====================================
def create_driver() -> webdriver.Chrome:
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--lang=ko-KR")
    driver = webdriver.Chrome(options=options)
    driver.implicitly_wait(2)
    return driver


# =====================================
# 보조 함수
# =====================================
def normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.replace("\n", " ").replace("\t", " ").replace("\r", " ")).strip()


def preview_text(text: str, limit: int = 120) -> str:
    text = normalize_whitespace(text)
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def get_text(element: WebElement) -> str:
    try:
        return normalize_whitespace(element.text)
    except (StaleElementReferenceException, WebDriverException):
        return ""


def find_first_clickable(
    driver: webdriver.Chrome,
    xpaths: Sequence[str],
    timeout: int = DEFAULT_WAIT_SECONDS,
) -> Optional[WebElement]:
    wait = WebDriverWait(driver, timeout)
    for xpath in xpaths:
        try:
            return wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
        except TimeoutException:
            continue
    return None


def parse_date(text: str) -> str:
    patterns = [
        r"(\d{4}-\d{1,2}-\d{1,2})",
        r"(\d{4}\.\d{1,2}\.\d{1,2})",
        r"(\d{4}/\d{1,2}/\d{1,2})",
        r"(\d{4}년\s*\d{1,2}월\s*\d{1,2}일)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "")
        if match:
            value = match.group(1)
            if "년" in value:
                value = value.replace("년", "-").replace("월", "-").replace("일", "")
                value = re.sub(r"\s+", "", value)
            return value.replace(".", "-").replace("/", "-")
    return ""


def check_date_range(date_str: str) -> str:
    """
    날짜 문자열을 받아 수집 범위 여부를 반환합니다.
    반환값:
        "valid" - START_DATE 이후 (수집)
        "old"   - START_DATE 이전 (수집 중단 신호)
        "error" - 파싱 실패 (스킵)
    """
    if not date_str:
        return "error"
    try:
        for fmt in ("%Y-%m-%d", "%Y-%M-%d"):
            try:
                date_obj = datetime.strptime(date_str, fmt)
                return "valid" if date_obj >= START_DATE else "old"
            except ValueError:
                continue
        return "error"
    except Exception:
        return "error"


def is_review_limit_reached(collected_dict: dict) -> bool:
    """MAX_REVIEWS 제한에 도달했는지 확인합니다."""
    if not MAX_REVIEWS:
        return False
    return len(collected_dict) >= MAX_REVIEWS


# =====================================
# 페이지 상단 정보 수집
# =====================================
def get_place_name(driver: webdriver.Chrome) -> str:
    """
    h2.tit_head에서 가게 이름을 추출합니다. (카카오맵 실측 확인)
    """
    selectors = [
        "h2.tit_head",
        ".tit_place",
        ".place_title",
        "h1",
    ]
    for selector in selectors:
        try:
            el = driver.find_element(By.CSS_SELECTOR, selector)
            name = normalize_whitespace(el.text)
            if name:
                print(f"[가게 이름] {name}  (selector: {selector})")
                return name
        except (WebDriverException, Exception):
            continue
    print("[WARN] 가게 이름을 찾지 못했습니다.")
    return ""


# =====================================
# 정렬 / 탭 클릭
# =====================================
def click_latest_sort(driver):
    try:
        btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, ".btn_sort"))
        )
        driver.execute_script("arguments[0].click();", btn)
        time.sleep(0.5)

        links = driver.find_elements(By.CSS_SELECTOR, ".link_sort")
        for link in links:
            if "최신" in link.text:
                driver.execute_script("arguments[0].click();", link)
                print("[최신순] 정렬 클릭 성공")
                time.sleep(1.0)
                return True

    except Exception as e:
        print(f"[DEBUG] click_latest_sort 예외: {e}")

    print("[WARN] 최신순 정렬 클릭 실패 - 기본 정렬로 진행")
    return False


# =====================================
# 리뷰 카드 수집
# =====================================
def is_valid_review_card(card: WebElement) -> bool:
    text = get_text(card)
    return bool(text) and bool(parse_date(text))


def get_review_cards(driver: webdriver.Chrome) -> List[WebElement]:
    selectors: Sequence[tuple] = [
        (By.CSS_SELECTOR, "[data-review-item]"),
        (By.CSS_SELECTOR, "li"),
        (By.CSS_SELECTOR, "article"),
    ]

    best_cards: List[WebElement] = []

    for by, selector in selectors:
        try:
            elements = driver.find_elements(by, selector)
            filtered = [element for element in elements if is_valid_review_card(element)]
            print(f"[DEBUG] selector={selector}, raw={len(elements)}, valid={len(filtered)}")

            if len(filtered) > len(best_cards):
                best_cards = filtered
        except WebDriverException as exc:
            print(f"[WARN] selector 실패: {selector} ({exc})")

    return best_cards


def make_review_key(row: dict) -> tuple:
    return (
        row.get("계정 ID", ""),
        row.get("방문 날짜", ""),
        row.get("리뷰 내용", ""),
    )


def expand_more_in_card(driver: webdriver.Chrome, card: WebElement) -> None:
    try:
        btn_more = card.find_element(By.CSS_SELECTOR, ".btn_more")
        driver.execute_script("arguments[0].click();", btn_more)
        time.sleep(0.3)
    except (WebDriverException, Exception):
        pass


def get_css_text(card: WebElement, selector: str) -> str:
    try:
        return normalize_whitespace(card.find_element(By.CSS_SELECTOR, selector).text)
    except (WebDriverException, Exception):
        return ""


def parse_one_card(driver: webdriver.Chrome, card: WebElement, idx: int) -> Optional[dict]:
    """
    리뷰 카드 1개를 파싱합니다.
    반환값:
        dict   - 정상 파싱된 리뷰 데이터
        "stop" - START_DATE 이전 리뷰 (수집 중단 신호, MAX_REVIEWS=None일 때만 활성화)
        None   - 스킵
    """
    # ── 0. 더보기 펼치기 ────────────────────────────────────────────────
    expand_more_in_card(driver, card)

    # ── 1. 닉네임 ────────────────────────────────────────────────────────
    account_id = ""
    try:
        name_el = card.find_element(By.CSS_SELECTOR, ".name_user")
        account_id = driver.execute_script(
            """
            const el = arguments[0].cloneNode(true);
            el.querySelectorAll('.screen_out').forEach(s => s.remove());
            return el.innerText.trim();
            """,
            name_el,
        )
    except (WebDriverException, Exception):
        pass

    # ── 2. 레벨 ──────────────────────────────────────────────────────────
    user_level = get_css_text(card, ".ico_badge")

    # ── 3. 리뷰 수 ───────────────────────────────────────────────────────
    account_review_count = ""
    raw_review_count = get_css_text(card, ".list_detail li:first-child")
    if raw_review_count:
        m = re.search(r"(\d+)", raw_review_count)
        if m:
            account_review_count = m.group(1)

    # ── 4. 리뷰어 별점 평균 ───────────────────────────────────────────────
    account_avg_rating = ""
    raw_avg = get_css_text(card, ".list_detail li:nth-child(2)")
    if raw_avg:
        m = re.search(r"(\d+\.?\d*)", raw_avg)
        if m:
            account_avg_rating = m.group(1)

    # ── 5. 날짜 ──────────────────────────────────────────────────────────
    visit_date = ""
    raw_date = get_css_text(card, ".txt_date")
    if raw_date:
        visit_date = parse_date(raw_date)

    # 날짜 범위 체크 (MAX_REVIEWS=None일 때만 중단 신호 반환)
    if MAX_REVIEWS is None:
        date_status = check_date_range(visit_date)
        if date_status == "old":
            return "stop"
        if date_status == "error":
            return None

    # ── 6. 별점 ──────────────────────────────────────────────────────────
    rating = ""
    try:
        sg_el = card.find_element(By.CSS_SELECTOR, ".starred_grade")
        rating = driver.execute_script(
            """
            const spans = arguments[0].querySelectorAll('.screen_out');
            return spans.length >= 2 ? spans[1].innerText.trim() : '';
            """,
            sg_el,
        )
    except (WebDriverException, Exception):
        pass

    # ── 7. 리뷰 본문 ─────────────────────────────────────────────────────
    review_text = ""
    try:
        desc_el = card.find_element(By.CSS_SELECTOR, ".desc_review")
        review_text = driver.execute_script(
            """
            const el = arguments[0].cloneNode(true);
            el.querySelectorAll('.btn_more, .btn_fold').forEach(s => s.remove());
            return el.innerText.trim();
            """,
            desc_el,
        )
        review_text = normalize_whitespace(review_text)
    except (WebDriverException, Exception):
        pass

    # 텍스트 없는 카드 제외 (별점만 남긴 유저)
    if not review_text:
        return None

    # ── 8. 사진 유무 ─────────────────────────────────────────────────────
    has_photo = False
    try:
        card.find_element(By.CSS_SELECTOR, ".review_thumb")
        has_photo = True
    except (WebDriverException, Exception):
        pass

    row = {
        "계정 ID":                    account_id,
        "계정의 리뷰 수":              account_review_count,
        "계정의 별점 평균":            account_avg_rating,
        "방문 날짜":                   visit_date,
        "별점":                        rating,
        "리뷰 내 사진 유무":            has_photo,
        "리뷰 글자 수":                len(review_text),
        "리뷰 내용":                   review_text,
        "카카오맵 리뷰다는 사람 레벨":  user_level,
    }

    return row


def collect_visible_reviews(driver, collected_dict, seen_ids: set) -> tuple:
    """
    현재 화면에 보이는 리뷰 카드를 수집합니다.
    반환값: (신규 수집 수, 중단 여부)
    """
    cards = get_review_cards(driver)
    new_count = 0
    parsed_count = 0
    should_stop = False

    for idx, card in enumerate(cards, start=1):
        if is_review_limit_reached(collected_dict):
            print(f"[INFO] 최대 수집 수({MAX_REVIEWS}개) 도달 → 수집 중단")
            should_stop = True
            break

        try:
            result = parse_one_card(driver, card, idx)

            if result == "stop":
                print(f"[INFO] START_DATE({START_DATE.strftime('%Y-%m-%d')}) 이전 리뷰 발견 → 수집 중단")
                should_stop = True
                break

            if result is None:
                continue

            # 닉네임 중복 체크
            account_id = result.get("계정 ID", "")
            if account_id and account_id in seen_ids:
                print(f"[SKIP] 중복 닉네임: {account_id}")
                continue
            if account_id:
                seen_ids.add(account_id)

            parsed_count += 1
            key = make_review_key(result)

            if key not in collected_dict:
                collected_dict[key] = result
                new_count += 1

                print("\n" + "=" * 70)
                print(f"[수집 #{len(collected_dict)}]", end="")
                if MAX_REVIEWS:
                    print(f"  (최대 {MAX_REVIEWS}개 중)")
                else:
                    print(f"  (기준일: {START_DATE.strftime('%Y-%m-%d')} 이후)")
                print(f"계정 ID        : {result['계정 ID']}")
                print(f"계정의 리뷰 수  : {result['계정의 리뷰 수']}")
                print(f"계정의 별점 평균: {result['계정의 별점 평균']}")
                print(f"방문 날짜       : {result['방문 날짜']}")
                print(f"별점            : {result['별점']}")
                print(f"사진 유무       : {result['리뷰 내 사진 유무']}")
                print(f"글자 수         : {result['리뷰 글자 수']}")
                print(f"리뷰 내용       : {preview_text(result['리뷰 내용'], 120)}")
                print("=" * 70)

                if is_review_limit_reached(collected_dict):
                    print(f"[INFO] 최대 수집 수({MAX_REVIEWS}개) 도달 → 수집 중단")
                    should_stop = True
                    break

        except Exception as e:
            print(f"[WARN] 카드 파싱 실패: {e}")

    print(
        f"[INFO] 파싱 성공: {parsed_count}개 / 신규: {new_count}개 / "
        f"누적: {len(collected_dict)}개",
        end=""
    )
    if MAX_REVIEWS:
        print(f" / 목표: {MAX_REVIEWS}개")
    else:
        print()

    return new_count, should_stop


# =====================================
# 스크롤 / 더보기 버튼
# =====================================
def try_click_more_button(driver: webdriver.Chrome) -> bool:
    xpaths = [
        "//*[self::a or self::button][contains(., '더보기')]",
        "//*[self::a or self::button][contains(., '후기 더보기')]",
        "//*[self::a or self::button][contains(., '리뷰 더보기')]",
    ]

    for xpath in xpaths:
        try:
            elements = driver.find_elements(By.XPATH, xpath)
            for el in elements:
                if el.is_displayed() and el.is_enabled():
                    driver.execute_script("arguments[0].click();", el)
                    print("[OK] 더보기 클릭")
                    time.sleep(1.5)
                    return True
        except WebDriverException:
            continue

    print("[WARN] 더보기 버튼 못 찾음")
    return False


def scroll_down(driver: webdriver.Chrome) -> None:
    try:
        current_height = driver.execute_script("return document.body.scrollHeight")
        step = current_height // 4
        for i in range(1, 5):
            driver.execute_script(f"window.scrollTo(0, {step * i});")
            time.sleep(0.4)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        print("[INFO] 단계적 스크롤 완료")
        time.sleep(3.0)
    except WebDriverException as e:
        print(f"[WARN] 스크롤 실패: {e}")


# =====================================
# 전체 수집 루프
# =====================================
def crawl_reviews() -> pd.DataFrame:
    driver = create_driver()

    try:
        print("[STEP] 페이지 접속 중...")
        driver.get(URL)
        time.sleep(PAGE_LOAD_SLEEP)

        place_name = get_place_name(driver)
        print(f"[가게 이름 수집 완료] {place_name}")  # TODO: 확인 후 제거

        latest_sort_ok = click_latest_sort(driver)
        print(f"latest_sort_ok={latest_sort_ok}")

        if MAX_REVIEWS:
            print(f"[INFO] 수집 모드: 최대 {MAX_REVIEWS}개")
        else:
            print(f"[INFO] 수집 모드: {START_DATE.strftime('%Y-%m-%d')} 이후 전체")

        collected_dict = {}
        seen_ids: set = set()  # 닉네임 중복 추적
        no_new_rounds = 0

        for round_idx in range(1, MAX_ROUNDS + 1):
            print(f"\n[===== 라운드 {round_idx}/{MAX_ROUNDS} =====]")

            new_count, stop = collect_visible_reviews(driver, collected_dict, seen_ids)

            if stop:
                break

            if MAX_REVIEWS is not None and len(collected_dict) >= MAX_REVIEWS:
                print(f"[INFO] 목표 {MAX_REVIEWS}개 도달 → 종료")
                break

            try_click_more_button(driver)
            scroll_down(driver)

            no_new_rounds = 0 if new_count > 0 else no_new_rounds + 1

            if no_new_rounds >= NO_NEW_LIMIT:
                print("[INFO] 더 이상 신규 리뷰 없음 → 종료")
                break

        rows = list(collected_dict.values())
        df = pd.DataFrame(rows)
        df.insert(0, "가게 이름", place_name)

        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

        print(f"\n[INFO] 총 수집 리뷰 수: {len(df)}개")
        print(f"[DONE] 저장 완료: {OUTPUT_CSV}")
        print(df.head(10))

        return df

    finally:
        driver.quit()


if __name__ == "__main__":
    crawl_reviews()