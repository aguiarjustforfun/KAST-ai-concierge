# KAST Concierge AI - Versão Potente e Estável (funcional em Windows)
# Carregamento lazy do modelo para evitar crashes no startup
# Autor: Grok para Tomás - Janeiro 2026

from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging
from datetime import datetime
from langdetect import detect, LangDetectException
import traceback
import os

# Logging para ver o que está a acontecer
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Rate limiting (segurança básica)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["100 per day", "20 per hour"]
)

# Variáveis globais lazy (carrega só quando necessário)
_model = None
_intent_cache = {}

def load_model():
    """Carrega o modelo apenas na primeira chamada (evita crash no import global)"""
    global _model
    if _model is None:
        logging.info("Carregando modelo de embeddings pela primeira vez...")
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logging.info("Modelo carregado com sucesso!")
        except Exception as e:
            logging.error(f"Erro ao carregar modelo: {e}")
            _model = None  # fallback para keyword matching
    return _model

def get_intent(query: str) -> str:
    """Detecta o intent com embeddings ou fallback simples"""
    query = query.lower().strip()
    
    # Fallback rápido se modelo não carregou
    if load_model() is None:
        keywords = {
            'depósito': ['depositar', 'depósito', 'tx hash', 'adicionar fundos'],
            'saldo': ['saldo', 'quanto tenho', 'balance'],
            'cartão': ['cartão', 'card', 'kard'],
            'fees': ['fees', 'taxas', 'custo', 'comissão'],
            'viagens': ['viagem', 'travel', 'fora país'],
            'suporte': ['ajuda', 'suporte', 'human', 'ticket'],
            'yield': ['yield', 'juros', 'apy', 'ganhar'],
            'cashback': ['cashback', 'recompensa', 'pontos']
        }
        for intent, words in keywords.items():
            if any(word in query for word in words):
                return intent
        return 'unknown'
    
    # Cache simples para evitar recalcular sempre
    if query in _intent_cache:
        return _intent_cache[query]
    
    try:
        from sentence_transformers import util
        import torch

        model = load_model()
        if model is None:
            return 'unknown'

        intents = ['depósito', 'saldo', 'cartão', 'fees', 'viagens', 'suporte', 'yield', 'cashback']
        query_emb = model.encode(query, convert_to_tensor=True)
        
        best_score = -1
        best_intent = 'unknown'
        
        for intent in intents:
            intent_emb = model.encode(intent, convert_to_tensor=True)
            score = util.cos_sim(query_emb, intent_emb).item()
            if score > best_score:
                best_score = score
                best_intent = intent
        
        if best_score > 0.62:  # threshold ajustado para mais precisão
            _intent_cache[query] = best_intent
            return best_intent
        
        _intent_cache[query] = 'unknown'
        return 'unknown'
    
    except Exception as e:
        logging.error(f"Erro no get_intent: {str(e)}")
        logging.error(traceback.format_exc())
        return 'unknown'

@app.route('/test')
def test_route():
    return "TESTE FUNCIONOU! Servidor KAST AI potente está online. 🚀"

@app.route('/greet/<name>')
def greet(name):
    return f"Olá {name}! Bem-vindo ao KAST Concierge AI. Como posso ajudar hoje?"

@app.route('/chat', methods=['POST'])
@limiter.limit("10 per minute")
def chat():
    data = request.get_json(silent=True) or {}
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({"response": "Escreve uma pergunta válida!"}), 400
    
    # Deteta o idioma da pergunta
    try:
        lang = detect(query)
    except LangDetectException:
        lang = 'pt'  # Se não detetar, usa português
    
    # Nome e saldo (podes mudar depois)
    name = "Tomás"
    balance = 1250.75
    
    # Respostas em vários idiomas
    responses = {
        'pt': {
            'greeting': f"Olá {name}! 👋 Saldo atual: {balance:.2f} USDC. ",
            'depósito': "Para depositar: App → Wallet → Deposit (USDC, SOL, etc.). Se tens tx hash, envia aqui!",
            'saldo': f"O teu saldo é {balance:.2f} USDC. Queres ver movimentos?",
            'cartão': "Cartão KAST ativo em 160+ países, sem taxas forex.",
            'fees': "Fees: 0% swaps internos, ~1% saques fiat, zero em viagens.",
            'viagens': "Perfeito para viagens: cartão global + Apple Pay.",
            'suporte': "Suporte humano: ticket no app ou support@kast.xyz",
            'yield': "Yield: até 4.5% APY em USDC (em breve).",
            'cashback': "Cashback: até 5-8% + pontos atuais 420.",
            'unknown': "Não percebi... Tenta reformular (ex: 'saldo', 'depositar', 'cartão')."
        },
        'en': {
            'greeting': f"Hi {name}! 👋 Current balance: {balance:.2f} USDC. ",
            'depósito': "To deposit: App → Wallet → Deposit (USDC, SOL, etc.). Send tx hash if you have one!",
            'saldo': f"Your balance is {balance:.2f} USDC. Want to see transactions?",
            'cartão': "KAST Card active in 160+ countries, no forex fees.",
            'fees': "Fees: 0% on internal swaps, ~1% on fiat withdrawals, zero on travel.",
            'viagens': "Perfect for travel: global card + Apple Pay.",
            'suporte': "Human support: open ticket in app or email support@kast.xyz",
            'yield': "Yield: up to 4.5% APY on USDC (coming soon).",
            'cashback': "Cashback: up to 5-8% + current points 420.",
            'unknown': "Didn't understand... Try rephrasing (e.g. 'balance', 'deposit', 'card')."
        },
        'es': {
            'greeting': f"¡Hola {name}! 👋 Saldo actual: {balance:.2f} USDC. ",
            'depósito': "Para depositar: App → Wallet → Deposit (USDC, SOL, etc.). ¡Envía tx hash si la tienes!",
            'saldo': f"Tu saldo es {balance:.2f} USDC. ¿Quieres ver movimientos?",
            'cartão': "Tarjeta KAST activa en +160 países, sin tasas forex.",
            'fees': "Comisiones: 0% en swaps internos, ~1% en retiros fiat, cero en viajes.",
            'viagens': "Perfecto para viajes: tarjeta global + Apple Pay.",
            'suporte': "Soporte humano: abre ticket en app o email support@kast.xyz",
            'yield': "Yield: hasta 4.5% APY en USDC (próximamente).",
            'cashback': "Cashback: hasta 5-8% + puntos actuales 420.",
            'unknown': "No entendí... Intenta reformular (ej: 'saldo', 'depositar', 'tarjeta')."
        },
        'de': {
            'greeting': f"Hallo {name}! 👋 Aktueller Saldo: {balance:.2f} USDC. ",
            'depósito': "Zum Einzahlen: App → Wallet → Deposit (USDC, SOL usw.). Sende tx hash, wenn du einen hast!",
            'saldo': f"Dein Saldo beträgt {balance:.2f} USDC. Möchtest du Transaktionen sehen?",
            'cartão': "KAST-Karte aktiv in über 160 Ländern, keine Forex-Gebühren.",
            'fees': "Gebühren: 0% bei internen Swaps, ~1% bei Fiat-Abhebungen, null bei Reisen.",
            'viagens': "Perfekt für Reisen: globale Karte + Apple Pay.",
            'suporte': "Menschlicher Support: Ticket in der App öffnen oder E-Mail an support@kast.xyz",
            'yield': "Yield: bis zu 4,5% APY auf USDC (kommt bald).",
            'cashback': "Cashback: bis zu 5-8% + aktuelle Punkte 420.",
            'unknown': "Nicht ganz verstanden... Versuche es umzuformulieren (z.B. 'Saldo', 'Einzahlen', 'Karte')."
        },
        'fr': {
            'greeting': f"Bonjour {name} ! 👋 Solde actuel : {balance:.2f} USDC. ",
            'depósito': "Pour déposer : App → Wallet → Deposit (USDC, SOL, etc.). Envoyez le tx hash si vous l'avez !",
            'saldo': f"Votre solde est de {balance:.2f} USDC. Voulez-vous voir les transactions ?",
            'cartão': "Carte KAST active dans plus de 160 pays, sans frais forex.",
            'fees': "Frais : 0 % sur les swaps internes, ~1 % sur les retraits fiat, zéro en voyage.",
            'viagens': "Parfait pour les voyages : carte globale + Apple Pay.",
            'suporte': "Support humain : ouvrez un ticket dans l'app ou envoyez un email à support@kast.xyz",
            'yield': "Yield : jusqu'à 4,5 % APY sur USDC (bientôt disponible).",
            'cashback': "Cashback : jusqu'à 5-8 % + points actuels 420.",
            'unknown': "Je n'ai pas bien compris... Essayez de reformuler (ex. : 'solde', 'déposer', 'carte')."
        },
        'it': {
            'greeting': f"Ciao {name}! 👋 Saldo attuale: {balance:.2f} USDC. ",
            'depósito': "Per depositare: App → Wallet → Deposit (USDC, SOL, ecc.). Invia tx hash se ce l'hai!",
            'saldo': f"Il tuo saldo è {balance:.2f} USDC. Vuoi vedere le transazioni?",
            'cartão': "Carta KAST attiva in oltre 160 paesi, senza commissioni forex.",
            'fees': "Commissioni: 0% su swap interni, ~1% su prelievi fiat, zero in viaggio.",
            'viagens': "Perfetto per i viaggi: carta globale + Apple Pay.",
            'suporte': "Supporto umano: apri un ticket nell'app o invia email a support@kast.xyz",
            'yield': "Yield: fino al 4,5% APY su USDC (in arrivo).",
            'cashback': "Cashback: fino al 5-8% + punti attuali 420.",
            'unknown': "Non ho capito bene... Prova a riformulare (es: 'saldo', 'depositare', 'carta')."
        },
        'nl': {
            'greeting': f"Hallo {name}! 👋 Huidig saldo: {balance:.2f} USDC. ",
            'depósito': "Om te storten: App → Wallet → Deposit (USDC, SOL, enz.). Stuur tx hash als je die hebt!",
            'saldo': f"Je saldo is {balance:.2f} USDC. Wil je transacties zien?",
            'cartão': "KAST-kaart actief in meer dan 160 landen, geen valutakosten.",
            'fees': "Kosten: 0% bij interne swaps, ~1% bij fiat-opnames, nul bij reizen.",
            'viagens': "Perfect voor reizen: globale kaart + Apple Pay.",
            'suporte': "Menselijke ondersteuning: open een ticket in de app of e-mail support@kast.xyz",
            'yield': "Yield: tot 4,5% APY op USDC (komt eraan).",
            'cashback': "Cashback: tot 5-8% + huidige punten 420.",
            'unknown': "Niet helemaal begrepen... Probeer het anders te formuleren (bijv. 'saldo', 'storten', 'kaart')."
        },
        'ru': {
            'greeting': f"Привет {name}! 👋 Текущий баланс: {balance:.2f} USDC. ",
            'depósito': "Чтобы пополнить: App → Wallet → Deposit (USDC, SOL и т.д.). Отправь tx hash, если есть!",
            'saldo': f"Твой баланс: {balance:.2f} USDC. Хочешь посмотреть транзакции?",
            'cartão': "Карта KAST активна в более 160 странах, без комиссий по обмену валют.",
            'fees': "Комиссии: 0% на внутренние свопы, ~1% на вывод в фиат, ноль в поездках.",
            'viagens': "Идеально для путешествий: глобальная карта + Apple Pay.",
            'suporte': "Человеческая поддержка: открой тикет в приложении или напиши support@kast.xyz",
            'yield': "Yield: до 4,5% APY на USDC (скоро).",
            'cashback': "Кэшбэк: до 5-8% + текущие баллы 420.",
            'unknown': "Не совсем понял... Попробуй перефразировать (например, 'баланс', 'пополнить', 'карта')."
        },
        'zh-cn': {
            'greeting': f"你好 {name}! 👋 当前余额：{balance:.2f} USDC。 ",
            'depósito': "存款方式：App → Wallet → Deposit（USDC、SOL 等）。如果你有 tx hash，请发送！",
            'saldo': f"你的余额是 {balance:.2f} USDC。想查看交易记录吗？",
            'cartão': "KAST 卡在 160+ 个国家/地区有效，无外汇费用。",
            'fees': "费用：内部兑换 0%，法币提现约 1%，旅行零费用。",
            'viagens': "非常适合旅行：全球卡 + Apple Pay。",
            'suporte': "人工支持：在应用中开票或邮件至 support@kast.xyz",
            'yield': "Yield：USDC 年化收益率高达 4.5%（即将推出）。",
            'cashback': "返现：高达 5-8% + 当前积分 420。",
            'unknown': "不太明白…请尝试重新表述（例如 '余额'、'存款'、'卡'）。"
        },
        'ja': {
            'greeting': f"こんにちは {name}！👋 現在の残高：{balance:.2f} USDC。 ",
            'depósito': "入金方法：App → Wallet → Deposit（USDC、SOL など）。tx hash がある場合は送ってください！",
            'saldo': f"あなたの残高は {balance:.2f} USDC です。取引履歴を見ますか？",
            'cartão': "KASTカードは160カ国以上で利用可能、為替手数料なし。",
            'fees': "手数料：内部スワップ 0%、法定通貨出金約1%、旅行中はゼロ。",
            'viagens': "旅行に最適：グローバルカード + Apple Pay。",
            'suporte': "人間サポート：アプリでチケットを開くか、support@kast.xyz にメール",
            'yield': "Yield：USDCで最大4.5% APY（近日公開）。",
            'cashback': "キャッシュバック：最大5-8% + 現在のポイント 420。",
            'unknown': "よくわかりませんでした…言い換えてみてください（例：'残高'、'入金'、'カード'）。"
        },
        'ko': {
            'greeting': f"안녕하세요 {name}! 👋 현재 잔액: {balance:.2f} USDC. ",
            'depósito': "입금 방법: App → Wallet → Deposit (USDC, SOL 등). tx hash가 있으면 보내주세요!",
            'saldo': f"잔액은 {balance:.2f} USDC입니다. 거래 내역을 보시겠습니까?",
            'cartão': "KAST 카드는 160개 이상 국가에서 사용 가능, 환전 수수료 없음.",
            'fees': "수수료: 내부 스왑 0%, 법정화폐 출금 약 1%, 여행 중 0.",
            'viagens': "여행에 최적: 글로벌 카드 + Apple Pay.",
            'suporte': "인간 지원: 앱에서 티켓 열기 또는 support@kast.xyz 로 이메일",
            'yield': "Yield: USDC 최대 4.5% APY (곧 출시).",
            'cashback': "캐시백: 최대 5-8% + 현재 포인트 420.",
            'unknown': "잘 이해하지 못했습니다… 다시 표현해 주세요 (예: '잔액', '입금', '카드')."
        },
        'ar': {
            'greeting': f"مرحبا {name}! 👋 الرصيد الحالي: {balance:.2f} USDC. ",
            'depósito': "للإيداع: App → Wallet → Deposit (USDC، SOL، إلخ). أرسل tx hash إذا كان لديك!",
            'saldo': f"رصيدك هو {balance:.2f} USDC. هل تريد رؤية المعاملات؟",
            'cartão': "بطاقة KAST نشطة في أكثر من 160 دولة، بدون رسوم تحويل عملة.",
            'fees': "الرسوم: 0% على المبادلات الداخلية، ~1% على سحب العملات الورقية، صفر في السفر.",
            'viagens': "مثالية للسفر: بطاقة عالمية + Apple Pay.",
            'suporte': "الدعم البشري: افتح تذكرة في التطبيق أو أرسل بريدًا إلى support@kast.xyz",
            'yield': "Yield: حتى 4.5% APY على USDC (قريبًا).",
            'cashback': "كاش باك: حتى 5-8% + نقاط حالية 420.",
            'unknown': "لم أفهم جيدًا... حاول إعادة الصياغة (مثال: 'الرصيد'، 'إيداع'، 'بطاقة')."
        },
        'sv': {
            'greeting': f"Hej {name}! 👋 Aktuell balans: {balance:.2f} USDC. ",
            'depósito': "För att sätta in: App → Wallet → Deposit (USDC, SOL osv.). Skicka tx hash om du har en!",
            'saldo': f"Din balans är {balance:.2f} USDC. Vill du se transaktioner?",
            'cartão': "KAST-kort aktivt i över 160 länder, inga valutaväxlingsavgifter.",
            'fees': "Avgifter: 0% på interna swaps, ~1% på fiat-uttag, noll på resor.",
            'viagens': "Perfekt för resor: globalt kort + Apple Pay.",
            'suporte': "Mänsklig support: öppna ett ärende i appen eller e-posta support@kast.xyz",
            'yield': "Yield: upp till 4,5% APY på USDC (kommer snart).",
            'cashback': "Cashback: upp till 5-8% + nuvarande poäng 420.",
            'unknown': "Förstod inte riktigt... Försök omformulera (t.ex. 'saldo', 'sätta in', 'kort')."
        },
        'pl': {
            'greeting': f"Cześć {name}! 👋 Aktualne saldo: {balance:.2f} USDC. ",
            'depósito': "Aby wpłacić: App → Wallet → Deposit (USDC, SOL itp.). Wyślij tx hash, jeśli masz!",
            'saldo': f"Twoje saldo to {balance:.2f} USDC. Chcesz zobaczyć transakcje?",
            'cartão': "Karta KAST aktywna w ponad 160 krajach, bez opłat za przewalutowanie.",
            'fees': "Opłaty: 0% na wewnętrzne swapy, ~1% na wypłaty fiat, zero w podróżach.",
            'viagens': "Idealna na podróże: karta globalna + Apple Pay.",
            'suporte': "Wsparcie ludzkie: otwórz zgłoszenie w aplikacji lub napisz na support@kast.xyz",
            'yield': "Yield: do 4,5% APY na USDC (wkrótce).",
            'cashback': "Cashback: do 5-8% + aktualne punkty 420.",
            'unknown': "Nie do końca zrozumiałem... Spróbuj inaczej sformułować (np. 'saldo', 'wpłacić', 'karta')."
        },
    }
    
    # Escolhe as respostas no idioma detetado (ou português se não souber)
    res = responses.get(lang, responses['pt'])
    
    intent = get_intent(query)
    
    # Monta a resposta
    resposta = res['greeting']
    if intent in res:
        resposta += res[intent]
    else:
        resposta += res['unknown']
    
    resposta += f"\n\n({datetime.now().strftime('%d/%m/%Y %H:%M')})"
    
    logging.info(f"Pergunta: '{query}' (idioma: {lang}) → Intent: {intent}")
    
    return jsonify({"response": resposta})

from solana.rpc.api import Client
from solders.signature import Signature
from datetime import datetime  
import logging 

# RPC público da Solana (mainnet – grátis)
SOLANA_RPC = "https://api.mainnet-beta.solana.com"

@app.route('/verify-tx', methods=['POST'])
def verify_tx():
    data = request.get_json(silent=True) or {}
    tx_hash = data.get('tx_hash', '').strip()
    
    if not tx_hash:
        return jsonify({"response": "Manda o tx hash! Exemplo: {'tx_hash': '5x...'}"}), 400
    
    try:
        client = Client(SOLANA_RPC)
        sig = Signature.from_string(tx_hash)
        tx = client.get_transaction(sig, max_supported_transaction_version=0)
        
        if tx.value is None:
            return jsonify({"response": "Transação não encontrada ou inválida."}), 404
        
        # Detalhes simples
        block_time = tx.value.block_time
        date_str = datetime.fromtimestamp(block_time).strftime("%d/%m/%Y %H:%M") if block_time else "Data desconhecida"
        
        # Mudança de saldo (simples – em SOL)
        meta = tx.value.transaction.meta
        pre_bal = meta.pre_balances[0] if meta and meta.pre_balances else 0
        post_bal = meta.post_balances[0] if meta and meta.post_balances else 0
        amount_changed = (post_bal - pre_bal) / 1_000_000_000 if pre_bal or post_bal else 0  # evita divisão por zero
        
        response = f"Transação válida! Data: {date_str}. Mudança de saldo: {amount_changed:.4f} SOL (aprox)."
        
        return jsonify({"response": response})
    
    except Exception as e:
        logging.error(f"Erro ao verificar tx Solana: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"response": f"Erro ao verificar: {str(e)}. Tenta outro tx hash."}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    print(f"Servidor Flask a correr em http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)