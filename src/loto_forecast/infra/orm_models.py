from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from loto_forecast.infra.db import Base


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(16), index=True)
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    resource_samples: Mapped[list[ResourceSample]] = relationship(
        back_populates="task",
        cascade="all, delete-orphan",
    )


class Model(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), index=True)
    version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    family: Mapped[str | None] = mapped_column(String(64), nullable=True)
    properties: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    hyperparams: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    evaluations: Mapped[list[Evaluation]] = relationship(
        back_populates="model",
        cascade="all, delete-orphan",
    )


class Evaluation(Base):
    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"), index=True)
    dataset_id: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)

    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifacts: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    analysis: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    model: Mapped[Model] = relationship(back_populates="evaluations")
    predictions: Mapped[list[PredictionRow]] = relationship(
        back_populates="evaluation",
        cascade="all, delete-orphan",
    )


class PredictionRow(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    evaluation_id: Mapped[int] = mapped_column(ForeignKey("evaluations.id"), index=True)
    t: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    y_true: Mapped[float] = mapped_column(Float)
    y_pred: Mapped[float] = mapped_column(Float)

    evaluation: Mapped[Evaluation] = relationship(back_populates="predictions")


class ResourceSample(Base):
    __tablename__ = "resource_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"), index=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    cpu_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    rss_mb: Mapped[float | None] = mapped_column(Float, nullable=True)
    vms_mb: Mapped[float | None] = mapped_column(Float, nullable=True)

    gpu_util: Mapped[float | None] = mapped_column(Float, nullable=True)
    gpu_mem_mb: Mapped[float | None] = mapped_column(Float, nullable=True)
    gpu_temp_c: Mapped[float | None] = mapped_column(Float, nullable=True)

    task: Mapped[Task] = relationship(back_populates="resource_samples")


Index("idx_resource_task_ts", ResourceSample.task_id, ResourceSample.ts)
