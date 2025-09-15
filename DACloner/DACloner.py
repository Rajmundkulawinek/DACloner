from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass
from dataclasses import dataclass, field
from langchain_core.messages import ToolMessage
from typing import List, Dict, Optional, Any, Set, Literal
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.agents import tool
from langgraph.types import Interrupt
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field
from datetime import datetime
from rich import print as rprint
from datetime import datetime, timezone
from langgraph.graph import StateGraph
from IPython.display import Image, display      
from langgraph.types import interrupt, Command


#----------------------------------------------------------------------------
#Gdzie w moim kodzie nie wykorzystuje w pełni mozliwości chatopenai - i popraw!
#Przejście na tryb streaming
#Przeanalizować dlaczego jest tak wolna
#----------------------------------------------------------------------------


import json
import re

# ---------------------------------------------------------------------------
# 1. Parameters
# ---------------------------------------------------------------------------
load_dotenv()  # wymaga OPENAI_API_KEY w .env
session_id = "terminal-user"
thread_id = "auto-fill-pydantic"
BASE_DIR = Path(__file__).parent
BASE_PROMPTS = BASE_DIR / ".." / "Prompts" / "Reusable"
GLOBAL_FILES = BASE_DIR / ".." / "Configs" 

# ---------------------------------------------------------------------------
# 2. Variables 
# ---------------------------------------------------------------------------
memory_store = {}


# ---------------------------------------------------------------------------
# 2. Files 
# ---------------------------------------------------------------------------

with open(BASE_PROMPTS / "Requirements_Colection.txt", encoding="utf-8") as f:
    REQUIREMENTS_COLECTION_PROMPT_TEXT = f.read().strip()
REQUIREMENTS_COLECTION_PROMPT_TEMPLATE = PromptTemplate.from_template(REQUIREMENTS_COLECTION_PROMPT_TEXT)

with open(BASE_PROMPTS / "Requirements_Colection_Detect_State.txt", encoding="utf-8") as f:
    REQUIREMENTS_COLECTION_PROMPT_Detect_State_TEXT = f.read().strip()
REQUIREMENTS_COLECTION_PROMPT_Detect_State_TEMPLATE = PromptTemplate.from_template(REQUIREMENTS_COLECTION_PROMPT_Detect_State_TEXT)

with open(BASE_PROMPTS / "Requirements_Analysis.txt", encoding="utf-8") as f:
    REQUIREMENTS_ANALYSIS_PROMPT_TEXT = f.read().strip()
REQUIREMENTS_ANALYSIS_PROMPT_TEMPLATE = PromptTemplate.from_template(REQUIREMENTS_ANALYSIS_PROMPT_TEXT)
# Wywołanie: REQUIREMENTS_ANALYSIS_PROMPT = REQUIREMENTS_ANALYSIS_PROMPT_TEMPLATE.format(table_list='xxx')

with open(BASE_PROMPTS / "Requirements_Analysis_Detect_State.txt", encoding="utf-8") as f:
    REQUIREMENTS_ANALYSIS_PROMPT_Detect_State_TEXT = f.read().strip()
REQUIREMENTS_ANALYSIS_PROMPT_Detect_State_TEMPLATE = PromptTemplate.from_template(REQUIREMENTS_ANALYSIS_PROMPT_Detect_State_TEXT)

with open(BASE_PROMPTS / "Point_Critical_Columns_In_Source.txt", encoding="utf-8") as f:
    Point_Critical_Columns_In_Source_Text = f.read().strip()
Point_Critical_Columns_In_Source_Template = PromptTemplate.from_template(Point_Critical_Columns_In_Source_Text)

#with open(BASE_PROMPTS / "Reusable_orchestrator.txt", encoding="utf-8") as f:
#    REUSABLE_ORCHESTRATOR_PROMPT = f.read().strip()

with open(BASE_PROMPTS / "Model_Dimension.txt", encoding="utf-8") as f:
    Model_Dimension_Text = f.read().strip()
Model_Dimension_Template = PromptTemplate.from_template(Model_Dimension_Text)

with open(BASE_PROMPTS / "Model_Dimension_Detect_State.txt", encoding="utf-8") as f:
    Model_Dimension_Detect_State_Text = f.read().strip()
Model_Dimension_Detect_State_Template = PromptTemplate.from_template(Model_Dimension_Detect_State_Text)

with open(BASE_PROMPTS / "Model_Dimension_finalization.txt", encoding="utf-8") as f:
    Model_Dimension_finalization_Text = f.read().strip()
Model_Dimension_finalization_Template = PromptTemplate.from_template(Model_Dimension_finalization_Text)


with open(GLOBAL_FILES / "List_available_dimensions.json", encoding="utf-8") as f:
    LIST_AVAILABLE_DIMENSIONS = f.read().strip()

with open(GLOBAL_FILES / "SAP_2_Snowflake_data_types_mapping.csv", encoding="utf-8") as f:
    SAP_2_Snowflake_data_types_mapping_file_content = f.read().strip()


# ---------------------------------------------------------------------------
# 2. LLM models
# ---------------------------------------------------------------------------
#llm_Reusable_orchestrator = ChatOpenAI(model_name="gpt-5-mini")
llm_Requirements_Colection = ChatOpenAI(model_name="gpt-5-mini")
llm_Requirements_Colection_Detect_State = ChatOpenAI(model_name="gpt-5-mini")
llm_Requirements_Analysis = ChatOpenAI(model_name="gpt-5-mini")
llm_Requirements_Analysis_Detect_State = ChatOpenAI(model_name="gpt-5-mini")
llm_Model_Dimension = ChatOpenAI(model_name="gpt-5-mini")
llm_Model_Dimension_Detect_State = ChatOpenAI(model_name="gpt-5-mini")
llm_Model_Dimension_finalization = ChatOpenAI(model_name="gpt-5-mini")
llm_Model_Fact = ChatOpenAI(model_name="gpt-5-mini")
llm_Model_Fact_Detect_State = ChatOpenAI(model_name="gpt-5-mini")
llm_point_critical_columns_in_source = ChatOpenAI(model_name="gpt-5-mini")

# ---------------------------------------------------------------------------
# 2. State 
# ---------------------------------------------------------------------------
class AvailableDimension(BaseModel):
    dimension_name:                 str  # mandatory
    main_source_table:              List[str]  # mandatory
    other_source_tables:            Optional[List[str]] = Field(default=None)
    business_key_on_source_side:    Optional[List[str]] = Field(default=None)
    surrogate_key:                  Optional[str] = Field(default=None)
    business_key_on_dimension_side: Optional[List[str]] = Field(default=None)

class AvailableDimensionEssence(BaseModel):
    dimension_name:                 str  # mandatory
    main_source_table:              List[str]  # mandatory

class ColumnDefinition(BaseModel):
    order:                          Optional[int] = Field(default=None, description="Columns order in dimension" )
    column_name:                    Optional[str] = Field(default=None, description="name of the collumn in target dimension" )
    column_source:                  Optional[str] = Field(default=None, description="name of the collumn in source table or calculation represent column value" )
    load_logic:                     Optional[str] = Field(default=None, description="Destcription how to load data in ETL process" )
    data_type:                      Optional[str] = Field(default=None, description="Column data type in final data warehouse" )
    length:                         Optional[int] = Field(default=None)
    precision:                      Optional[int] = Field(default=None)
    column_comment:                 Optional[str] = Field(default=None, description="Comment give light on busines meaining of keep values" )
    nullable:                       Optional[bool] = Field(default=False)
    PII:                            Optional[bool] = Field(default=False,  description="yes - if field contains personal informations" )
    column_confidentiality_level:   Optional[str] = Field(default=None, description="Data confidentiality clasyfication: C1 - Highly Confidential, C2 - Confidential, C3 - Internal, C4 - Public")
    PK:                             Optional[bool] = Field(default=False, description="Information if field is primary key")
    UK:                             Optional[bool] = Field(default=False, description="Information if field is part of unique key")
    FK:                             Optional[str] = Field(default=None, description="Information if field is foreign key and to which [table].[column] is referring")


class FactToCreate(BaseModel):
    fact_name:              str  # mandatory
    main_source_table:      List[str]  # mandatory
    other_source_tables:    Optional[List[str]] = Field(default=None)
    connected_dimensions:   Optional[List[str]] = Field(default=None)
    created:                Optional[bool] = Field(default=False)
    column_list:            Optional[List[ColumnDefinition]] = Field(default=None)

class DimensionToCreate(BaseModel):
    dimension_name:                         str  # mandatory
    dimension_comment:                      Optional[str]  = Field(default=None, description="Business description of dimension with information about main source table" )
    main_source_table:                      List[str]  # mandatory
    business_key_on_source_side:            Optional[List[str]] = Field(default=None, description="Point which column from source table is the busines key" )
    other_source_tables:                    Optional[List[str]] = Field(default=None, description="Point other tables which need to build compex dimension" )
    surrogate_key:                          Optional[str] = Field(default=None, description="Point surrodate key in final dim. Usually dim_<name>_key" )
    business_key_on_dimension_side:         Optional[List[str]] = Field(default=None, description="Point columns which are representing busines key on dimension side" )    
    detailed_column_list:                   Optional[List[ColumnDefinition]] = Field(default=None, description="List of details informations about final dimension columns. For each column it is seperate object" )
    critical_columns_analyze_txt:           Optional[str] = Field(default=None, description="Sugestia którą kolumnę należy uwzględnić w wymiarze i dlaczego" )
    source_tables_analyze_txt:              Optional[str] = Field(default=None, description="Analizy na poziomie kolumn wszystkich tabel źródłowych dostarczone przez SAP eksperta lub analityka" )    
    design_approved:                        bool = Field(default=False, description="Information if user approved designe for this dimension" )
    ddl:                                    Optional[str] = Field(default=None, description="Definition of DDL script" )
    sql:                                    Optional[str] = Field(default=None, description="Definition of SQL script" )
    model_definition:                       Optional[str] = Field(default=None, description="Definition of model which will feed designe tool" )
    scripts_approved:                       bool = Field(default=False, description="Scripts: ddl, sql, model_definition has approved by user" )
    
# MAIN class: DesignState
class DesignState(BaseModel):
    required_source_tables:             List[str] = Field( default_factory=list, description="Tabele źródłowe przekazane przez użytkownika do analizy" )
    additional_source_tables:           Optional[List[str]] = Field( default_factory=list, description="Tabele źródłowe zaproponowane i zaakceptowane w wyniku analizy" )
    required_source_tables_approved:    bool = Field( default=False, description="Czy user zaakceptował inicjalną listę tabel żródłowych?" )
    additional_source_tables_approved:  bool = Field( default=False, description="Czy user zaakceptował dodatkową listę tabel żródłowych- zaproponowaną mu na czacie?" )
    objects_to_create_approved :        bool = Field( default=False, description="Czy user zaakceptował listę faktów / wymiarów do zamodelowania?")
    last_user_message:                  Optional[str] = Field(default=None)
    last_10_messages:                   List[str] = Field( default_factory=list, description="Ostatnie 10 wiadomości (user + AI)" )
    available_dimensions:               List[AvailableDimension] = Field(default_factory=list, description="Szczegółowe informacje o obiektach już istniejących w whurtowni" )
    available_dimensions_essence:       List[AvailableDimensionEssence] = Field(default_factory=list, description="Najważniejsze informacje o obiektach już istniejących w whurtowni" )
    facts_to_create:                    List[FactToCreate] = Field(default_factory=list, description="Lista faktów do stworzenia wraz z ich szczegółami" )
    dimensions_to_create:               List[DimensionToCreate] = Field(default_factory=list, description="Lista wymiarów do stworzenia wraz z ich szczegółami" )
    currently_modeled_object:           Optional[str] = Field(default=None, description="Fact or dimension which is actually modeled" ) 
    source_table_analyze:               Optional[dict[str, str]] = Field(default_factory=dict, description="Analiza tabeli źródłowej dostarczona przez SAP eksperta lub analityka na bazie wskazanego pliku" )
    SAP_2_Snowflake_data_types:         Optional[str] = None
    awaiting_input_for:                 Optional[Literal[ "requirements_colection", "requirements_analysis", "model_dimension" ]] = None
# ---------------------------------------------------------------------------
# 3. Helpers functions
# ---------------------------------------------------------------------------

def update_state_with_whitelist(state, updates: dict, allowed: set[str]):
    """
    Aktualizuje tylko pola z białej listy.
    - ignoruje resztę
    - nie nadpisuje None/pustymi (oprócz bool)
    Zwraca zaktualizowany state.
    """
    for key, value in updates.items():
        if key not in allowed:
            continue
        if isinstance(value, bool) or (value not in (None, "", [], {})):
            setattr(state, key, value)
    return state

def sanitize_message_for_ui(s: str) -> str:
    """Usuń niechciane JSON-owe escape'y przed wysłaniem do UI."""
    if not isinstance(s, str):
        return str(s)

    # jeśli mamy string opakowany cudzysłowami → spróbuj odczytać jako JSON
    if len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:
        try:
            return json.loads(s)
        except Exception:
            pass

    # zamień uciekające sekwencje na realne znaki
    return s.replace("\\n", "\n").replace("\\t", "\t")

def _is_nonempty_value(value: Any) -> bool:
    """
    True, jeżeli wartość jest 'znana' i niepusta.
    - None -> False (nie aktualizujemy)
    - ""   -> False
    - []   -> False
    - dla bool -> zawsze True (jeśli klucz wystąpił, to świadomy update, nawet False)
    - dla liczb/strukt. -> True, jeśli nie None i niepuste (dla str/list)
    """
    if isinstance(value, bool):
        return True  # bool ma być aktualizowany, jeśli klucz istnieje
    if value is None:
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    if isinstance(value, (list, tuple)) and len(value) == 0:
        return False
    return True

def _known_fields_of_dimension_to_create() -> Set[str]:
    try:
        # Pydantic v2
        return set(DimensionToCreate.model_fields.keys())
    except Exception:
        # Pydantic v1
        return set(DimensionToCreate.__fields__.keys())

def _filter_known_fields(
    d: Dict[str, Any],
    include_empty: Optional[Set[str]] = None,
    allowed_fields: Optional[Set[str]] = None
) -> Dict[str, Any]:
    """
    Zwraca tylko znane pola z DimensionToCreate, pomijając puste/nieznane.
    Możesz dodatkowo ograniczyć do 'allowed_fields'.
    """
    include_empty = include_empty or set()
    known = _known_fields_of_dimension_to_create()
    out = {}
    for k, v in d.items():
        if k not in known:
            continue
        if allowed_fields is not None and k not in allowed_fields:
            continue
        if k in include_empty:
            out[k] = v
        else:
            if _is_nonempty_value(v):
                out[k] = v
    return out

def _update_dimension_in_place(target: DimensionToCreate, patch: Dict[str, Any]) -> None:
    """
    Aktualizuje istniejący obiekt DimensionToCreate tylko wartościami niepustymi.
    Boole są aktualizowane, jeśli klucz jest obecny (nawet jeśli False).
    """
    known = _known_fields_of_dimension_to_create()
    for k, v in patch.items():
        if k not in known:
            continue
        # Dla boola: aktualizuj zawsze, skoro klucz jest w patchu
        if isinstance(getattr(target, k, None), bool) or isinstance(v, bool):
            setattr(target, k, v)
            continue
        # Dla innych typów: tylko wartości 'niepuste'
        if _is_nonempty_value(v):
            setattr(target, k, v)

def load_SAP_2_Snowflake_data_types_mapping_to_state (state: DesignState) -> DesignState:
    """ Wczytuje mapowanie z pliku SAP_2_Snowflake_data_types_mapping do obiektu DesignState. """

    # Ustawienie danych w stanie
    state.SAP_2_Snowflake_data_types = SAP_2_Snowflake_data_types_mapping_file_content

    return state

def load_list_available_dimensions_to_state (state: DesignState) -> DesignState:
    """ Wczytuje dostępne wymiary z pliku LIST_AVAILABLE_DIMENSIONS do obiektu DesignState. """
    # Parsowanie JSON-a ze stringa
    raw_data = json.loads(LIST_AVAILABLE_DIMENSIONS)

    # Zamiana listy słowników na listę obiektów AvailableDimensions
    parsed_dimensions: List[AvailableDimension] = [
        AvailableDimension(**item) for item in raw_data
    ]
    
    # Ustawienie danych w stanie
    state.available_dimensions = parsed_dimensions

    # Przepisywanie wartości z available_dimensions do parsed_essence_dimensions:
    parsed_essence_dimensions: List[AvailableDimensionEssence] = [
        AvailableDimensionEssence(
            dimension_name=dim.dimension_name,
            main_source_table=dim.main_source_table
        )
        for dim in parsed_dimensions
    ]

    # Ustawienie wersji "essence" w stanie
    state.available_dimensions_essence = parsed_essence_dimensions

    return state

def requirements_colection_update_state (state: DesignState) -> DesignState:
    """ LLM generuje jsona z required_source_tables i required_source_tables_approved na podstawie dialogu z użytkownikiem a reszta funkcji wpycha to do stanu """

    ALLOWED_STATE_CHANGES = {
        "required_source_tables",
        "required_source_tables_approved"
    }
    messages = [
        SystemMessage(content=REQUIREMENTS_COLECTION_PROMPT_Detect_State_TEXT),
        HumanMessage(content=f"Odpowiedź użytkownika: {state.last_10_messages}")
    ]
    response = llm_Requirements_Colection_Detect_State.invoke(messages).content.strip()
    try:
        parsed = json.loads(response)
        detected = {k: v for k, v in parsed.items() if v is not None}
        state = update_state_with_whitelist(state, detected, ALLOWED_STATE_CHANGES)
    except Exception:
        print("Debug - issue in requirements_colection_update_state")

def requirements_analysis_update_state (state: DesignState) -> DesignState:
    """ LLM generuje jsona z extended_source_tables, facts required_source_tables_approved na podstawie dialogu z użytkownikiem a reszta funkcji wpycha to do stanu """
    ALLOWED_STATE_CHANGES = {
        "facts_to_create"
        "fact_name",
        "main_source_table",
        "other_source_tables",
        "connected_dimensions",
        "dimensions_to_create",
        "dimension_name",
        "main_source_table",
        "business_key_on_source_side",
        "other_source_tables",
        "surrogate_key",
        "business_key_on_dimension_side",
        "additional_source_tables",
        "additional_source_tables_approved",
        "objects_to_create_approved"
    }
    messages = [
        SystemMessage(content=REQUIREMENTS_ANALYSIS_PROMPT_Detect_State_TEXT),
        HumanMessage(content=f"Odpowiedź użytkownika: {state.last_10_messages}")
    ]
    response = llm_Requirements_Analysis_Detect_State.invoke(messages).content.strip()
    try:
        parsed = json.loads(response)
        detected = {k: v for k, v in parsed.items() if v is not None}
        state = update_state_with_whitelist(state, detected, ALLOWED_STATE_CHANGES)
    except Exception:
        print('Debug - issue in requirements_analysis_update_state')
    return state

def model_dimension_update_state (state: DesignState) -> DesignState:
    """
    1) Skupia się wyłącznie na DesignState.dimensions_to_create.
    2) Aktualizuje tylko znane elementy/pola, nie nadpisuje pustymi/nieznanymi.
    3) Dodaje nowe DimensionToCreate tylko jeśli ma obowiązkowe:
       - dimension_name (str, niepusty)
       - main_source_table (lista, niepusta)
    4) Pozostałych atrybutów DesignState nie dotyka.
    """

    ALLOWED_STATE_CHANGES = {
       "dimension_name",                         
        "dimension_comment",                      
        "main_source_table",                      
        "business_key_on_source_side",            
        "other_source_tables",                    
        "surrogate_key",                          
        "business_key_on_dimension_side",         
        "detailed_column_list",                  
        #"critical_columns_analyze_txt",           
        #"source_tables_analyze_txt",              
        "design_approved",                        
        "ddl",                                    
        "sql",                                    
        "model_definition",                       
        "scripts_approved",                   
        "order",                          
        "column_name",                    
        "column_source",                  
        "load_logic",                     
        "data_type",                      
        "length",                         
        "precision",                      
        "column_comment",                 
        "nullable",                       
        "PII"
    } 
    messages = [
        SystemMessage(content=Model_Dimension_Detect_State_Text),
        HumanMessage(content=f"Odpowiedź użytkownika (last_10_messages): {state.last_10_messages} , Aktualny stan wymiaru: {dimension_as_string(state, state.currently_modeled_object)} ")
    ]
    response = llm_Model_Dimension_Detect_State.invoke(messages).content.strip()
    try:
        parsed = json.loads(response) if response else {}
    except Exception:
        print('322! Wtopa... nie udało się sparsować JSONa i model_dimension_update_state nie uaktualnia stanu')
        return state

    # Oczekujemy listy pod kluczem 'dimensions_to_create'
    items = parsed.get("dimensions_to_create")
    if not isinstance(items, list) or len(items) == 0:
        # Nic do zrobienia (albo LLM nie rozpoznał)
        print('329! Wtopa... niema dimensions_to_create ')
        return state

    # Indeks istniejących wymiarów po nazwie (case-insensitive dla wygody)
    existing_by_name = {
        (d.dimension_name or "").strip().lower(): d
        for d in state.dimensions_to_create
        if isinstance(d, DimensionToCreate) and (d.dimension_name or "").strip() != ""
    }

    for raw_item in items:
        if not isinstance(raw_item, dict):
            continue

        # Bierzemy tylko znane pola i odfiltrowujemy puste oraz niedozowolne
        patch = _filter_known_fields(raw_item,ALLOWED_STATE_CHANGES)

        dim_name = (patch.get("dimension_name") or raw_item.get("dimension_name") or "").strip()
        if not dim_name:
            # Bez nazwy nie wiemy czego dotyczy — pomijamy
            continue

        key = dim_name.lower()
        target = existing_by_name.get(key)

        if target is not None:
            # Update istniejącego wymiaru
            _update_dimension_in_place(target, patch)
        else:
            # Dodanie nowego wymiaru tylko jeśli są mandatory i niepuste
            main_src = patch.get("main_source_table") or raw_item.get("main_source_table")
            if isinstance(main_src, list) and len(main_src) > 0:
                # Tworzymy obiekt, ale nadal filtrujemy puste pola
                to_create = _filter_known_fields(raw_item,ALLOWED_STATE_CHANGES)
                # Upewnijmy się, że nazwa i main_source_table są obecne i poprawne
                to_create["dimension_name"] = dim_name
                to_create["main_source_table"] = main_src
                try:
                    new_dim = DimensionToCreate(**to_create)
                    state.dimensions_to_create.append(new_dim)
                    existing_by_name[key] = new_dim
                except Exception:
                    # Jeśli walidacja Pydantic się wywali — ignorujemy ten wpis
                    print ("389 walidacja Pydantic się wywaliła — ignorujemy ten wpis ",to_create )
                    continue
            else:
                # Brakuje obowiązkowych danych — nie tworzymy
                continue
    #print ("376 Do stanu udało się wpakować: ",state.dimensions_to_create[state.currently_modeled_object] )
    return state

def dimension_as_string(state: DesignState, name: str) -> str:
    dim = next((d for d in state.dimensions_to_create if d.dimension_name == name), None)
    if dim is None:
        raise ValueError(f"Nie znaleziono wymiaru: {name}")

    # Pydantic v2
    if hasattr(dim, "model_dump_json"):
        return dim.model_dump_json(indent=2, exclude_none=True)

    # Pydantic v1 (fallback)
    return dim.json(indent=2, exclude_none=True)

def point_critical_columns_in_source(dimension_name: str, source_tables: str, source_tables_analyze_txt: str) -> str:   

    messages = [
        SystemMessage(content=Point_Critical_Columns_In_Source_Text, ),
        HumanMessage(content=(
                f"Modelujesz wymiar (dimension_name): {dimension_name}\n"
                f"Tabele źródłowe to (source_tables): {source_tables}\n"
                f"Przeanalizuj kolumny z tabel źródłowych. Jeżeli user uploadował dodatkową analizę, znajdziesz ją tu (source_tables_analyze_txt). Uwaga - nie pomijaj w analizie żadnej z kolumn: {source_tables_analyze_txt}\n"
            ))
    ]
    response = llm_point_critical_columns_in_source.invoke(messages).content.strip()
    return response

# ---------------------------------------------------------------------------
# 3. Nodes
# ---------------------------------------------------------------------------
# --- ROUTER (NODE) ---
def router_node(state: DesignState, config):
    return {}  # zawsze diff

# --- ROUTER (DECIDER) ---
def router_decider(state: DesignState):
    # Po pierwsze - wróć do węzła jeżeli czeka na odpowiedź
    if state.awaiting_input_for:
        return state.awaiting_input_for  # hard return to the same node
    if not ( state.required_source_tables and state.required_source_tables_approved ):
        return "requirements_colection"
    if ( not(state.facts_to_create or state.dimensions_to_create) or not state.objects_to_create_approved):
        return "requirements_analysis"
    if any(dim.design_approved is False for dim in state.dimensions_to_create):
        return "model_dimension"
    #if not (getattr(state, "model_fact_approved", False) and state.facts_to_create):
    #    return "model_fact"
    return END

def node_requirements_colection(state: DesignState, config) -> DesignState | Interrupt:
    session_id = config["configurable"]["session_id"]
    thread_id = config["configurable"]["thread_id"]
    memory_key = f"{session_id}:{thread_id}"
    # Historia rozmowy
    if memory_key not in memory_store:
        memory_store[memory_key] = ChatMessageHistory()
    history = memory_store[memory_key]
    # Dodaj wiadomość użytkownika do historii
    if state.last_user_message:
        history.add_user_message(state.last_user_message)
        state.last_10_messages.append(f"🧑 {state.last_user_message}")
        requirements_colection_update_state(state)
        state.last_user_message = None  # Wyczyść po interpretacji
        state.awaiting_input_for = None # Wyczyść po interpretacji

    # EARLY EXIT – jeśli już mamy komplet, nie generuj nowej wypowiedzi. Zabezpieczenie
    if state.required_source_tables and state.required_source_tables_approved:
        return state

    # Zbuduj prompt z obecnym stanem
    current_data = {
        "required_source_tables": state.required_source_tables,
        "required_source_tables_approved": state.required_source_tables_approved,
        "last_10_messages": state.last_10_messages
    }
    messages = REQUIREMENTS_COLECTION_PROMPT_TEMPLATE.format(current_data=json.dumps(current_data, ensure_ascii=False, indent=2))

    reply = llm_Requirements_Colection.invoke(messages).content.strip()
    history.add_ai_message(reply)
    state.last_10_messages.append(f"🤖 {reply}")
    state.last_10_messages = state.last_10_messages[-10:]  # Trzymaj tylko 10 ostatnich

    requirements_colection_update_state(state)

    # Czy wszystko co potrzebujemy zostało zebrane?
    if not (state.required_source_tables and state.required_source_tables_approved):
        state.awaiting_input_for = "requirements_colection"
        return interrupt({
            "message": sanitize_message_for_ui(reply),
            "next_state": state
        })
        
    return state

def node_requirements_analysis(state: DesignState, config) -> DesignState | Interrupt:
    session_id = config["configurable"]["session_id"]
    thread_id = config["configurable"]["thread_id"]
    memory_key = f"{session_id}:{thread_id}"
    # Historia rozmowy
    if memory_key not in memory_store:
        memory_store[memory_key] = ChatMessageHistory()
    history = memory_store[memory_key]
    # Dodaj wiadomość użytkownika do historii
    if state.last_user_message:
        history.add_user_message(state.last_user_message)
        state.last_10_messages.append(f"🧑 {state.last_user_message}")
        requirements_analysis_update_state(state)
        state.last_user_message = None  # Wyczyść po interpretacji
        state.awaiting_input_for = None # Wyczyść po interpretacji

    # EARLY EXIT – nic nie mów, jeśli etap już zamknięty
    if (state.facts_to_create or state.dimensions_to_create) and state.objects_to_create_approved:
        return state
    
    def to_plain(obj):
        if isinstance(obj, BaseModel):
            return obj.model_dump()            # Pydantic v2
        if isinstance(obj, list):
            return [to_plain(x) for x in obj]
        if isinstance(obj, dict):
            return {k: to_plain(v) for k, v in obj.items()}
        return obj  

    # Zbuduj prompt z obecnym stanem
    current_data = {
        "required_source_tables": state.required_source_tables,
        "additional_source_tables": state.additional_source_tables ,
        "available_dimensions_essence": state.available_dimensions_essence,
        "facts_to_create": state.facts_to_create,
        "dimensions_to_create": state.dimensions_to_create,
        "additional_source_tables_approved": state.additional_source_tables_approved ,    
        "objects_to_create_approved": state.objects_to_create_approved,
        "last_10_messages": state.last_10_messages,
        "source_table_analyze": state.source_table_analyze
    }

    current_data_plain = to_plain(current_data)  # <-- KONWERSJA
    human_msg = HumanMessage(
        content="Stan wejściowy:\n" + json.dumps(current_data_plain, ensure_ascii=False, indent=2)
    )
  
    system_msg = SystemMessage(content=REQUIREMENTS_ANALYSIS_PROMPT_TEXT)
    reply = llm_Requirements_Analysis.invoke([system_msg, human_msg]).content.strip()
  
    history.add_ai_message(reply)
    state.last_10_messages.append(f"🤖 {reply}")
    state.last_10_messages = state.last_10_messages[-10:]  # Trzymaj tylko 10 ostatnich

    requirements_analysis_update_state(state)

    # Czy wszystko co potrzebujemy zostało zebrane?
    if not ((state.facts_to_create or state.dimensions_to_create) and state.objects_to_create_approved):
        state.awaiting_input_for = "requirements_analysis"
        return interrupt({"message": sanitize_message_for_ui(reply), "next_state": state})
    state.awaiting_input_for = None
    return state
   
def node_model_dimension(state: DesignState, config) -> DesignState | Interrupt:
    session_id = config["configurable"]["session_id"]
    thread_id = config["configurable"]["thread_id"]
    memory_key = f"{session_id}:{thread_id}"
    # Historia rozmowy
    if memory_key not in memory_store:
        memory_store[memory_key] = ChatMessageHistory()
    history = memory_store[memory_key]
    # Dodaj wiadomość użytkownika do historii
    if state.last_user_message:
        history.add_user_message(state.last_user_message)
        state.last_10_messages.append(f"🧑 {state.last_user_message}")
        state.last_10_messages = state.last_10_messages[-10:]  # Trzymaj tylko 10 ostatnich
        model_dimension_update_state(state)
        state.last_user_message = None  # Wyczyść po interpretacji
        state.awaiting_input_for = None # Wyczyść po interpretacji

    # 
    #
    #   Jak nie posiadasz kluczy w analize, Potwierdź klucze je na leanix! Zdażały się przekłamania !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! TO DO!
    #
    #

    # EARLY EXIT – nic nie mów, jeśli etap już zamknięty
    if all(( dim.design_approved is True and dim.scripts_approved is True )  for dim in state.dimensions_to_create):
        state.currently_modeled_object=None
        return state


    for dimension in [                          #dla każdego wymiaru który ma cos jeszcze do zrobienia
        d for d in state.dimensions_to_create 
        if (not d.scripts_approved or not d.design_approved)
    ]:      
        # Zanim zaczniesz modelować nastepny wymiar, sprawdź czy nie ma czegoś zamodelowanego ale bez DDL SQL i feedu do toola modelującego
        if (dimension.design_approved and not dimension.scripts_approved):
            state.currently_modeled_object=dimension.dimension_name
            dimension_state_txt = dimension_as_string(state, state.currently_modeled_object)
            human_msg = HumanMessage(content=(
                f"Generuj ddl, sql oraz model_definition dla: {dimension.dimension_name}\n"
                f"Na podstawie dialogu: {state.last_10_messages}\n"
                f"Oraz aktualnego stan wymiaru: \n{dimension_state_txt}\n"
            ))
            system_msg = SystemMessage(
                content=Model_Dimension_finalization_Text
            )
            reply = llm_Model_Dimension_finalization.invoke([system_msg, human_msg]).content.strip()
            history.add_ai_message(reply)
            state.last_10_messages.append(f"🤖 {reply}")
            state.last_10_messages = state.last_10_messages[-10:]  # Trzymaj tylko 10 ostatnich
            model_dimension_update_state(state)
            #print('Powinnismy mieć gotowe skrypty', dimension)
            #dimension.scripts_approved=True
            return interrupt({"message": sanitize_message_for_ui(reply), "next_state": state})          

        if not dimension.design_approved:
            # Znalazliśmy pierwszy wymiar, którego design nie jest zaakceptowany
            state.currently_modeled_object=dimension.dimension_name
            # Sprawdzamy jego wszystkie tabele źródłowe
            source_tables = (dimension.main_source_table or []) + (dimension.other_source_tables or [])
            
            #Sprawdzenie czy mamy analizę krytycznych kolumn
            if not (dimension.critical_columns_analyze_txt and dimension.critical_columns_analyze_txt.strip()):
                # Zbuduj mapę: lower_key -> oryginalny_klucz
                key_map = {str(k).strip().lower(): k for k in state.source_table_analyze.keys()}

                parts = []
                for table in source_tables:
                    t_norm = str(table).strip().lower()
                    if t_norm in key_map:
                        real_key = key_map[t_norm]                     # faktyczny klucz w dict
                        val = state.source_table_analyze.get(real_key) # tekst analizy
                        if val:                                        # pomiń puste
                            parts.append(f"### {table} table analyze: \n{val}")

                dimension.source_tables_analyze_txt = "\n\n".join(parts)

                dimension.critical_columns_analyze_txt = point_critical_columns_in_source(
                    dimension.dimension_name,
                    source_tables,
                    dimension.source_tables_analyze_txt
                )
            
            dimension_state_txt = dimension_as_string(state, state.currently_modeled_object) 
            human_msg = HumanMessage(content=(
                f"Modeluj wymiar: {state.currently_modeled_object}\n"
                f"Na podstawie dialogu: {state.last_10_messages}\n"
                f"Aktualny stan wymiaru: \n{dimension_state_txt}\n"
            ))
            system_msg = SystemMessage(
                content=Model_Dimension_Text
            )
            
            reply = llm_Model_Dimension.invoke([system_msg, human_msg]).content.strip()
            history.add_ai_message(reply)
            state.last_10_messages.append(f"🤖 {reply}")
            state.last_10_messages = state.last_10_messages[-10:]  # Trzymaj tylko 10 ostatnich
            #model_dimension_update_state(state)
            if any(dim.design_approved is False for dim in state.dimensions_to_create) or any(dim.scripts_approved is False for dim in state.dimensions_to_create) :
                state.awaiting_input_for = "model_dimension"
                return interrupt({"message": sanitize_message_for_ui(reply), "next_state": state})
                

       # break  # Przerywamy, ponieważ znaleźliśmy pierwszy niezaakceptowany wymiar i zebraliśmy informacje o jego tabelach źródłowych
    return state


# ---------------------------------------------------------------------------
# 5. Graph
# ---------------------------------------------------------------------------
checkpointer = InMemorySaver()
graph = StateGraph(state_schema=DesignState)

graph.add_node("router", router_node)  # <- NODE
graph.add_node("requirements_colection", node_requirements_colection)
graph.add_node("requirements_analysis", node_requirements_analysis)
graph.add_node("model_dimension", node_model_dimension)
# graph.add_node("model_fact", node_model_fact)

graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",
    router_decider,  # <- DECIDER
    {
        "requirements_colection": "requirements_colection",
        "requirements_analysis": "requirements_analysis",
        "model_dimension": "model_dimension",
        # "model_fact": "model_fact",
        END: END,
    },
)

graph.add_edge("requirements_colection", "router")
graph.add_edge("requirements_analysis", "router")
graph.add_edge("model_dimension", "router")
# graph.add_edge("model_fact", "router")

runnable = graph.compile(checkpointer=InMemorySaver())

# ---------------------------------------------------------------------------
# 4. Initial
# ---------------------------------------------------------------------------





def run_cli():
    rprint("[bold green] Let's get to work! (CLI) [/bold green]")
    state = DesignState()
    load_list_available_dimensions_to_state(state)
    load_SAP_2_Snowflake_data_types_mapping_to_state(state)

    while True:
        result = runnable.invoke(
            state,
            config={"configurable": {"session_id": session_id, "thread_id": thread_id}}
        )
        if "__interrupt__" in result:
            interrupt_obj = result["__interrupt__"][0]
            msg = interrupt_obj.value["message"]
            state = interrupt_obj.value["next_state"]
            rprint(f"\n🤖 [cyan]LLM:[/cyan] {msg}")
            last_user_message = input("🧑 Ty: ")
            state.last_user_message = last_user_message
        else:
            state = result
            rprint(state)
            break


if __name__ == "__main__":
    run_cli()
