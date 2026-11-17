# Project Atlas · AWS RAG

[English](README.md) · [마이그레이션 사례](docs/aws-portfolio.ko.md) · [벤치마크 근거](benchmarks/atlas-aws21/summary.json) · [GitHub 저장소 생성 안내](docs/github-setup.ko.md)

**검색은 AWS에서, 답변 생성은 로컬에서 수행하는 한·영 기술 프로그램 코파일럿입니다.** Databricks Track A에서 AWS로 검색 계층을 바꾼 뒤, **고정 평가 50문항 모두에서 Top 5 근거 순서와 최종 답변 텍스트가 일치**했습니다.

| 기존 벤치마크 검증 항목 | 결과 |
| --- | --- |
| Top 5 순서 일치 / 답변 텍스트 일치 | 각각 50/50문항 |
| 검색 품질 | Recall@5 **0.9800**, MRR **0.8557** 유지 |
| 인용 라벨 | 50/50문항에서 유효. 답변의 사실 정확도와는 별개 |
| 아키텍처 | Lambda와 S3에서 검색, 맥에서 BGE-M3 임베딩과 Ollama `qwen3:8b` 생성 |

Lambda가 비공개 S3 스냅샷을 읽고 벡터 검색, 식별자 기반 BM25·RRF, 선택적 재정렬, 다중 식별자 근거 보완을 수행합니다. 반환된 근거를 바탕으로 로컬 모델이 한·영 답변과 출처를 생성합니다. 모델까지 AWS에 호스팅하는 구조는 아닙니다.

## 빠른 실행

**AWS 자격증명, 별도 패키지 설치, 모델 다운로드 없이 공개 검색 데모를 실행**할 수 있습니다. Python 3.12로 저장소 루트에서 실행하세요.

```bash
python3 examples/offline_demo.py
python3 aws/atlas-aws21/tests/test_lambda_contract.py
```

데모는 모의 S3, 합성 문서, 고정 테스트 벡터를 사용해 실제 Lambda 핸들러를 실행합니다. `identifier_hybrid` 경로, 두 이슈의 직접 근거, `Coverage complete: True`를 확인할 수 있습니다. 실제 AWS 호출이나 모델 임베딩, 답변 생성은 하지 않습니다.

## AWS 검색과 로컬 답변 생성 데모

실제 답변까지 확인하려면 AWS CLI로 호출할 수 있는 Atlas Lambda가 필요합니다. 로컬 의존성을 설치하고 Ollama에서 `qwen3:8b`를 사용할 수 있게 준비하세요.

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements-local.txt
```

**공개 합성 스냅샷**으로 만든 배포를 테스트할 때는 아래 질문을 사용합니다. 함수 이름이나 리전이 기본값과 다르면 `--function`, `--region`을 지정하세요.

```bash
./.venv/bin/python scripts/query_rag_aws_v04_v021.py \
  --query "DEMO-101과 DEMO-202의 현재 상태는 각각 무엇인가?"
```

배포용 ZIP을 만들거나 합성 문서를 BGE-M3로 임베딩하는 방법은 [AWS 배포 가이드](aws/atlas-aws21/README.md)에 있습니다. 기존 스냅샷으로 배포한 Lambda를 이미 운영 중이라면 재배포 없이 원래 `AUTH-342`와 `OBS-119` 질문을 사용할 수 있습니다.

## 저장소 구성

| 경로 | 내용 |
| --- | --- |
| `aws/atlas-aws21/` | Lambda 소스, ZIP 빌더, IAM 예시, 오프라인 테스트 |
| `examples/` | 공개 합성 문서와 오프라인 검색 데모 |
| `src/`, `config/`, `scripts/` | 로컬 임베딩, 프롬프트, Ollama 생성, 출처 검사, 실제 호출 데모 |
| `benchmarks/atlas-aws21/` | 문항별 일치 여부 50건을 포함한 공개용 평가 결과 |
| `benchmarks/historical/` | 기존 환경에서 실행하는 원본 평가 스크립트와 재현 조건 |
| `docs/` | 한·영 사례 문서와 GitHub 저장소 생성 가이드 |

공개된 수치는 **원본 청크 128개와 고정 50문항**에서 얻었습니다. 합성 데모는 이와 별개이며 해당 점수를 재현하지 않습니다. 원본 문서, 벡터, 평가 질문 원문, 답변 로그는 포함하지 않았습니다. 두 환경이 공통으로 놓친 사례와 평가의 한계는 [마이그레이션 사례](docs/aws-portfolio.ko.md)에 설명했습니다.
