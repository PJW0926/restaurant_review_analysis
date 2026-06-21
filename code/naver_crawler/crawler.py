from pydoc import html
import re
import time
import pandas as pd

from datetime import datetime

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from config import (
    SLEEP, MAX_IDLE_ROUNDS, SAFETY_MAX_ROUNDS,
    TEMP_CSV, START_DATE, MAX_REVIEWS
)


# =====================================
# 드라이버 생성
# =====================================
def create_driver():
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(options=options)  # Selenium 4.6+ 자동으로 드라이버 관리
    driver.implicitly_wait(2)
    return driver


# =====================================
# 보조 함수
# =====================================
def safe_text(parent, by, selector, default=""):
    try:
        return parent.find_element(by, selector).text.strip()
    except Exception:
        return default


def exists(parent, by, selector):
    try:
        parent.find_element(by, selector)
        return True
    except Exception:
        return False


def normalize_date_string(y, m, d):
    return f"{int(y):04d}.{int(m):02d}.{int(d):02d}"


def extract_visit_date_from_text(text):
    m1 = re.search(r"(\d{4})\.(\d{1,2})\.(\d{1,2})", text)
    if m1:
        y, m, d = m1.groups()
        return normalize_date_string(y, m, d)

    m2 = re.search(r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일", text)
    if m2:
        y, m, d = m2.groups()
        return normalize_date_string(y, m, d)

    return ""


def check_date_range(date_str):
    """
    날짜 문자열을 받아 수집 범위 여부를 반환합니다.
    반환값:
        "valid" - START_DATE 이후 (수집)
        "old"   - START_DATE 이전 (수집 중단 신호)
        "error" - 파싱 실패 (스킵)
    """
    try:
        date_obj = datetime.strptime(date_str, "%Y.%m.%d")
        return "valid" if date_obj >= START_DATE else "old"
    except Exception:
        return "error"


def parse_visit_info_from_text(full_text):
    visit_date = extract_visit_date_from_text(full_text)

    visit_count = ""
    count_match = re.search(r"(\d+번째 방문)", full_text)
    if count_match:
        visit_count = count_match.group(1)

    auth_method = ""
    for candidate in ["영수증", "예약", "주문", "포장", "배달"]:
        if candidate in full_text:
            auth_method = candidate
            break

    return visit_date, visit_count, auth_method


def extract_account_review_count(full_text):
    match = re.search(r"리뷰\s*([\d,]+)", full_text)
    if match:
        return match.group(1).replace(",", "")
    return ""


def count_review_chars(review_text):
    return len(review_text or "")


def preview_text(text, max_len=120):
    text = (text or "").strip()
    return text if len(text) <= max_len else text[:max_len] + "..."


def make_review_key(row):
    return (
        row.get("계정 ID", "").strip(),
        row.get("방문 날짜", "").strip(),
        row.get("리뷰 내용", "").strip()
    )


def get_place_name(driver):
    """페이지에서 식당 이름을 가져옵니다. 실패 시 빈 문자열 반환."""
    for selector in [
        "span.GHAhO",
        "h2.place_name",
        "h1.place_name",
        "[class*='place_name']",
        "[class*='placeName']",
    ]:
        try:
            name = driver.find_element(By.CSS_SELECTOR, selector).text.strip()
            if name:
                print(f"[INFO] 식당 이름 수집: {name}")
                return name
        except Exception:
            continue
    print("[WARN] 식당 이름을 찾지 못했습니다.")
    return ""



def extract_review_text_by_xpath(driver, idx):
    """XPath로 리뷰 텍스트를 직접 가져옵니다. <br> 태그는 공백으로 처리합니다."""
    xpath = f'//*[@id="_review_list"]/li[{idx}]/div[5]/a[1]'
    try:
        elem = driver.find_element(By.XPATH, xpath)
        html = elem.get_attribute("innerHTML") or ""
        html = re.sub(r"<br\s*/?>", " ", html, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", html)
        text = text.replace("&nbsp;", " ")
        text = re.sub(r"[\r\n\t]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception:
        return ""


# =====================================
# iframe 진입 / 리뷰 탭 클릭
# =====================================
def switch_to_entry_iframe(driver):
    driver.switch_to.default_content()
    for iframe_id in ["entryIframe", "searchIframe"]:
        try:
            WebDriverWait(driver, 5).until(
                EC.frame_to_be_available_and_switch_to_it((By.ID, iframe_id))
            )
            print(f"[iframe] 진입 성공: {iframe_id}")
            return
        except TimeoutException:
            continue
    raise Exception("[ERROR] iframe을 찾지 못했습니다.")


def click_review_tab(driver):
    for xpath in [
        "//a[contains(., '리뷰')]",
        "//button[contains(., '리뷰')]",
        "//span[contains(., '리뷰')]"
    ]:
        try:
            elem = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, xpath))
            )
            driver.execute_script("arguments[0].click();", elem)
            time.sleep(2)
            print("[리뷰 탭] 클릭 성공")
            return
        except Exception:
            continue
    print("[WARN] 리뷰 탭 클릭 실패 - 이미 리뷰 탭일 수도 있음")


# =====================================
# 스크롤 / 더보기 버튼
# =====================================
def get_review_scroll_box(driver):
    for selector in [
        "div[role='main']",
        "div.place_section_content",
        "div.place_section",
        "div#_pcmap_list_scroll_container",
        "div[class*='review']",
        "body"
    ]:
        try:
            return driver.find_element(By.CSS_SELECTOR, selector)
        except Exception:
            continue
    return driver.find_element(By.TAG_NAME, "body")


def click_more_buttons(driver):
    total_clicked = 0
    for xpath in [
        "//a[contains(., '펼쳐서 더보기')]",
        "//button[contains(., '펼쳐서 더보기')]",
        "//a[contains(., '더보기')]",
        "//button[contains(., '더보기')]"
    ]:
        try:
            for btn in driver.find_elements(By.XPATH, xpath):
                if btn.is_displayed() and btn.is_enabled():
                    driver.execute_script("arguments[0].click();", btn)
                    time.sleep(0.15)
                    total_clicked += 1
        except Exception:
            pass
    return total_clicked


def scroll_once(driver):
    scroll_box = get_review_scroll_box(driver)
    before = driver.execute_script("return arguments[0].scrollTop;", scroll_box)
    height = driver.execute_script("return arguments[0].scrollHeight;", scroll_box)
    driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight;", scroll_box)
    time.sleep(SLEEP)
    after = driver.execute_script("return arguments[0].scrollTop;", scroll_box)
    return before, after, height


# =====================================
# 리뷰 카드 파싱
# =====================================
def get_review_cards(driver):
    for selector in [
        "#_review_list > li",
        "li.EjjAW",
        "li.place_apply_pui",
        "div[class*='place_apply_pui']",
        "li[class*='pui']"
    ]:
        cards = driver.find_elements(By.CSS_SELECTOR, selector)
        if cards:
            print(f"[DEBUG] selector={selector} / count={len(cards)}")
            return cards
    return []


def parse_one_card(driver, card, idx, place_name=""):
    """
    리뷰 카드 1개를 파싱합니다.
    반환값:
        dict  - 정상 파싱된 리뷰 데이터
        "stop" - START_DATE 이전 리뷰 (수집 중단 신호)
        None  - 스킵 (날짜 없음, 범위 초과 등)
    """
    full_text = card.text.strip()
    if not full_text:
        return None

    account_id = safe_text(
        card, By.CSS_SELECTOR,
        ".pui__NMi-Dp, .pui__uslU0d, .place_bluelink, [class*='nick'], [class*='name']"
    )
    if not account_id:
        lines = [line.strip() for line in full_text.split("\n") if line.strip()]
        if lines:
            account_id = lines[0]

    account_review_count = extract_account_review_count(full_text)
    visit_date, visit_count, auth_method = parse_visit_info_from_text(full_text)

    # 날짜 범위 체크
    date_status = check_date_range(visit_date)
    if date_status == "old":
        return "stop"   # START_DATE 이전 → 수집 중단
    if date_status == "error":
        return None     # 파싱 실패 → 스킵

    review_text = extract_review_text_by_xpath(driver, idx)

# 카드를 화면 중앙으로 스크롤해서 lazy load 트리거
    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", card)
    time.sleep(0.5)  # 사진 로드 대기

    html = card.get_attribute("innerHTML") or ""
    has_photo = bool(re.search(r'pup-review-phinf\.pstatic\.net/[A-Za-z0-9]', html))

    
    row = {
        "식당 이름":       place_name,
        "계정 ID":        account_id,
        "계정의 리뷰 수":  account_review_count,
        "방문 날짜":       visit_date,
        "방문 횟수":       visit_count,
        "인증 수단":       auth_method,
        "리뷰 내용":       review_text,
        "리뷰 글자 수":    count_review_chars(review_text),
        "리뷰 내 사진 유무": has_photo
    }

    if not row["계정 ID"] and not row["리뷰 내용"]:
        return None

    return row


def is_review_limit_reached(collected_dict):
    """MAX_REVIEWS 제한에 도달했는지 확인합니다."""
    if not MAX_REVIEWS:
        return False
    return len(collected_dict) >= MAX_REVIEWS


def collect_visible_reviews(driver, collected_dict, place_name=""):
    """
    현재 화면에 보이는 리뷰 카드를 수집합니다.
    반환값: (신규 수집 수, 중단 여부)
    """
    cards = get_review_cards(driver)
    new_count = 0
    should_stop = False

    for idx, card in enumerate(cards, start=1):
        # MAX_REVIEWS 도달 시 즉시 중단
        if is_review_limit_reached(collected_dict):
            print(f"[INFO] 최대 수집 수({MAX_REVIEWS}개) 도달 → 수집 중단")
            should_stop = True
            break

        try:
            result = parse_one_card(driver, card, idx, place_name)

            if result == "stop":
                print("[INFO] START_DATE 이전 리뷰 발견 → 수집 중단")
                should_stop = True
                break

            if result is None:
                continue

            key = make_review_key(result)
            if key not in collected_dict:
                collected_dict[key] = result
                new_count += 1

                print("\n" + "=" * 70)
                print(f"[수집 #{len(collected_dict)}]", end="")
                if MAX_REVIEWS:
                    print(f"  (최대 {MAX_REVIEWS}개 중)")
                else:
                    print()
                print(f"계정 ID        : {result['계정 ID']}")
                print(f"방문 날짜      : {result['방문 날짜']}")
                print(f"인증 수단      : {result['인증 수단']}")
                print(f"글자 수        : {result['리뷰 글자 수']}")
                print(f"리뷰 내용      : {preview_text(result['리뷰 내용'])}")
                print("=" * 70)

                # 추가 후 즉시 재확인
                if is_review_limit_reached(collected_dict):
                    print(f"[INFO] 최대 수집 수({MAX_REVIEWS}개) 도달 → 수집 중단")
                    should_stop = True
                    break

        except Exception as e:
            print(f"[WARN] 카드 파싱 실패: {e}")

    print(f"[INFO] 신규 {new_count}개 / 누적 {len(collected_dict)}개", end="")
    if MAX_REVIEWS:
        print(f" / 목표 {MAX_REVIEWS}개")
    else:
        print()
    return new_count, should_stop


# =====================================
# 전체 수집 루프
# =====================================
def collect_all_reviews(driver):
    place_name = get_place_name(driver)  # 식당 이름 먼저 수집
    collected = {}
    idle_rounds = 0

    if MAX_REVIEWS:
        print(f"[INFO] 최대 수집 수: {MAX_REVIEWS}개")

    for round_idx in range(1, SAFETY_MAX_ROUNDS + 1):
        print(f"\n[===== 라운드 {round_idx} =====]")

        clicked = click_more_buttons(driver)
        if clicked:
            print(f"[INFO] 더보기 클릭: {clicked}회")

        new_count, stop = collect_visible_reviews(driver, collected, place_name)
        if stop:
            break

        if collected:
            pd.DataFrame(list(collected.values())).to_csv(
                TEMP_CSV, index=False, encoding="utf-8-sig"
            )

        idle_rounds = 0 if new_count > 0 else idle_rounds + 1

        before, after, height = scroll_once(driver)
        print(f"[INFO] scrollTop: {before} → {after} / height={height}")

        click_more_buttons(driver)
        new_count2, stop2 = collect_visible_reviews(driver, collected, place_name)
        if stop2:
            break

        if collected:
            pd.DataFrame(list(collected.values())).to_csv(
                TEMP_CSV, index=False, encoding="utf-8-sig"
            )

        idle_rounds = 0 if new_count2 > 0 else idle_rounds + 1

        if idle_rounds >= MAX_IDLE_ROUNDS:
            print("[INFO] 신규 리뷰 없음 → 수집 종료")
            break

    return list(collected.values())