abstract type AbstractNode end

mutable struct Node{T} <: AbstractNode
    parent::Union{Node{T},Nothing}
    children::Union{Vector{Node{T}}, Nothing}
    data::T

    Node{T}() where T = new{K}(nothing, nothing, nothing)
    Node{T}(data::T) where T = new{T}(nothing, nothing, nothing, data)
end
