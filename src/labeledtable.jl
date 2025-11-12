# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    LabeledTable(table, names)

Stores a Tables.jl `table` along with column `names` that
identify which columns are labels for supervised learning.
"""
struct LabeledTable{T}
  table::T
  labels::Vector{Symbol}
end

function LabeledTable(table, names)
  Tables.istable(table) || throw(ArgumentError("please provide a valid Tables.jl table"))
  cols = Tables.columns(table)
  vars = Tables.columnnames(cols)
  labs = selector(names)(vars)
  labs ⊆ vars || throw(ArgumentError("all labels must be column names in the table"))
  vars ⊆ labs && throw(ArgumentError("there must be at least one feature column in the table"))
  LabeledTable{typeof(table)}(table, labs)
end

Tables.istable(::Type{<:LabeledTable}) = true

Tables.rowaccess(::Type{<:LabeledTable{T}}) where {T} = Tables.rowaccess(T)

Tables.columnaccess(::Type{<:LabeledTable{T}}) where {T} = Tables.columnaccess(T)

Tables.rows(t::LabeledTable) = Tables.rows(t.table)

Tables.columns(t::LabeledTable) = Tables.columns(t.table)

Tables.columnnames(t::LabeledTable) = Tables.columnnames(t.table)

# -----------
# IO METHODS
# -----------

function Base.summary(io::IO, t::LabeledTable)
  name = nameof(typeof(t))
  nlab =  length(t.labels)
  print(io, "$name with $nlab label(s)")
end

Base.show(io::IO, t::LabeledTable) = summary(io, t)

function Base.show(io::IO, ::MIME"text/plain", t::LabeledTable)
  pretty_table(io, t; backend=:text, _common_kwargs(t)...)
end

function Base.show(io::IO, ::MIME"text/html", t::LabeledTable)
  pretty_table(
    io,
    t;
    backend=:html,
    _common_kwargs(t)...,
    renderer=:show,
    style=HtmlTableStyle(title=["font-size" => "14px"])
  )
end

function _common_kwargs(t)
  cols = Tables.columns(t)
  vars = Tables.columnnames(cols)

  labels = map(vars) do var
    if var ∈ t.labels
      styled"{(weight=bold),magenta:$var}"
    else
      styled"{(weight=bold):$var}"
    end
  end

  (
    title=summary(t),
    column_labels=collect(labels),
    maximum_number_of_rows=10,
    new_line_at_end=false,
    alignment=:c
  )
end

"""
    label(table, names)

Creates a `LabeledTable` from `table` using `names` as label columns.
"""
label(table, names) = LabeledTable(table, names)
