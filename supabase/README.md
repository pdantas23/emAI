# supabase/

Migrations e configuração de banco do emAI.

A fonte da verdade do schema continua sendo `src/storage/models.py` (SQLModel).
Os arquivos aqui são a versão SQL explícita — com `CHECK`, RLS, índices e
comentários — que a gente aplica em produção (Supabase) e que serve como
documento de review.

```
supabase/
└── migrations/
    └── 20260416120000_init_processed_emails.sql   # tabela + RLS + policies
```

## Como aplicar

### Via Supabase CLI (recomendado)

```bash
# primeira vez, linka o projeto local ao project-ref do Supabase
supabase link --project-ref <seu-project-ref>

# aplica tudo que ainda não foi aplicado
supabase db push
```

### Via psql (self-hosted / Supabase direto)

```bash
psql "$DATABASE_URL" -f supabase/migrations/20260416120000_init_processed_emails.sql
```

### Via SQL Editor do Supabase

Cole o conteúdo do arquivo `.sql` no **SQL Editor** e execute. A migração
é idempotente (`CREATE TABLE IF NOT EXISTS`, `CREATE POLICY` uma vez só),
então rodar duas vezes é seguro — o segundo run vai falhar no `CREATE POLICY`
porque policies não suportam `IF NOT EXISTS` no Postgres 15. Nesse caso,
dropar as policies antes ou rodar o rollback no rodapé do arquivo e aplicar
de novo.

## Policies (RLS)

A tabela `processed_emails` tem **RLS habilitado + FORCE**. O backend do emAI
conecta com a `service_role` key (que ignora RLS). Qualquer cliente usando
`anon` ou `authenticated` vai receber zero linhas — é por design, o conteúdo
dos emails é confidencial.

Se um dia o emAI virar multi-tenant, troca as policies RESTRICTIVE por
PERMISSIVE com `USING (owner_id = auth.uid())` e adiciona a coluna `owner_id`.

## CHECK constraints

O banco NÃO confia na camada Python: `priority` e `delivery_status` têm
`CHECK` que rejeitam valores fora dos enums. A constraint
`processed_emails_summary_matches_status_chk` é o invariante mais importante —
garante que:

- `delivered`          ⇒ os 3 campos de summary estão preenchidos E `relevance=TRUE`
- `skipped_irrelevant` ⇒ os 3 campos de summary são NULL E `relevance=FALSE`

Isso fecha a porta pra um bug futuro onde alguém persistisse uma linha
"delivered sem summary" ou "irrelevante com resumo".
