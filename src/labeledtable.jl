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

"""
    label(table, names)

Creates a `LabeledTable` from `table` using `names` as label columns.
"""
label(table, names) = LabeledTable(table, names)
