from pipeline.router import QueryRoute, RuleBasedQueryRouter


def test_router_short_circuits_greeting_without_rag():
    decision = RuleBasedQueryRouter().route("hello")

    assert decision.route == QueryRoute.GREETING
    assert decision.should_run_rag is False
    assert decision.refused is False


def test_router_refuses_out_of_scope_query_without_rag():
    decision = RuleBasedQueryRouter().route("What is the weather tomorrow?")

    assert decision.route == QueryRoute.OUT_OF_SCOPE
    assert decision.should_run_rag is False
    assert decision.refused is True


def test_router_keeps_research_question_on_rag_path():
    decision = RuleBasedQueryRouter().route(
        "What does the FlashAttention paper say about IO complexity?"
    )

    assert decision.route == QueryRoute.RAG_FACTUAL
    assert decision.should_run_rag is True


def test_router_identifies_follow_up_shape_for_rewriting_path():
    decision = RuleBasedQueryRouter().route("What about its limitations?")

    assert decision.route == QueryRoute.FOLLOW_UP
    assert decision.should_run_rag is True


def test_router_identifies_summary_and_exact_keyword_queries():
    router = RuleBasedQueryRouter()

    assert router.route("Summarize the main idea").route == QueryRoute.RAG_SUMMARY
    assert router.route("What is the F1 score in table 2?").route == (
        QueryRoute.RAG_EXACT_KEYWORD_TABLE
    )
