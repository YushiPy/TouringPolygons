from datetime import datetime, timezone
from sqlmodel import Field, Relationship, SQLModel


def get_utctime() -> datetime:
    return datetime.now(timezone.utc)


class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    password_hash: str

    drawings: list[Drawing] = Relationship(back_populates="user")


class Drawing(SQLModel, table=True):

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")

    data: str  # JSON string containing the drawing data

    created_at: datetime = Field(default_factory=get_utctime)
    modified_at: datetime = Field(default_factory=get_utctime)

    user: User | None = Relationship(back_populates="drawings")
