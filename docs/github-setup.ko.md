# 새 GitHub 저장소 만들기

권장 저장소 이름은 **`project-atlas-aws-rag`**입니다. 프로젝트 이름, AWS, RAG가 한눈에 보이고 기존 Databricks 프로젝트와도 구분됩니다. 이 ZIP은 기존 프로젝트에 덧붙이는 패키지가 아닙니다. 압축을 풀면 바로 새 저장소로 초기화할 수 있는 폴더가 나옵니다.

## 1. GitHub 웹에서 빈 저장소 생성

[새 저장소 만들기](https://github.com/new)에서 아래 값을 사용합니다.

| 항목 | 값 |
| --- | --- |
| Owner | 본인의 GitHub 계정 |
| Repository name | `project-atlas-aws-rag` |
| Description | `Bilingual technical program copilot with AWS Lambda retrieval, local Ollama generation, and a 50-query migration benchmark.` |
| Visibility | 바로 포트폴리오로 공유하려면 **Public**. 검토 후 공개하려면 **Private**. |
| Add README / .gitignore / License | 초기 생성에서 추가하지 않음. README와 .gitignore는 ZIP에 포함됨. 라이선스는 결정 후 별도로 추가. |

`Create repository`를 누르고 Quick Setup 화면의 HTTPS URL을 복사합니다. GitHub에서 README를 미리 만들지 않으면 로컬에서 만든 첫 커밋을 그대로 올릴 수 있습니다.

## 2. 맥에서 압축 해제 후 첫 커밋

Finder에서 ZIP을 풉니다. 아래 명령은 폴더가 다운로드 폴더에 그대로 있다고 가정합니다.

```bash
cd "$HOME/Downloads/project-atlas-aws-rag"
git init -b main
git add -- .gitignore README.md README.ko.md requirements-local.txt \
  aws benchmarks config docs examples scripts src
git diff --cached --check
git status --short
git commit -m "Add Project Atlas AWS RAG portfolio"
```

## 3. GitHub URL 연결 후 첫 푸시

아래 `YOUR_GITHUB_USERNAME`을 본인 계정으로 바꾸거나, 복사한 HTTPS URL을 통째로 넣습니다.

```bash
git remote add origin "https://github.com/YOUR_GITHUB_USERNAME/project-atlas-aws-rag.git"
git remote -v
git push -u origin main
```

## 4. GitHub에서 확인

- 첫 화면에 영문 README가 표시되고, **한국어** 링크가 열리는지 확인합니다.
- 한·영 포트폴리오와 50문항 결과 링크가 열리는지 확인합니다.
- 기존 원본 데이터나 실행 로그가 포함되지 않았는지 확인합니다.

**AWS 설정은 변경하지 않습니다.** 이 과정은 GitHub 게시를 위한 것으로, 기존 Lambda 재배포나 평가 재실행이 필요하지 않습니다.

참고 [GitHub 공식 안내 · 로컬 코드를 GitHub에 추가하기](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github?platform=mac)
