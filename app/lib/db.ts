import { openDB } from 'idb';
import type { DBSchema, IDBPDatabase } from 'idb';

interface ResumeRecord {
	id?: number;
	username: string;
	filename?: string;
	file?: Blob | null;
	jobTitle: string;
	score: number;
	createdAt: number;
}

interface ResumeDB extends DBSchema {
	resumes: {
		key: number;
		value: ResumeRecord;
		indexes: { 'by-username': string };
	};
}

let dbPromise: Promise<IDBPDatabase<ResumeDB>> | null = null;

export function initDB() {
	if (!dbPromise) {
		dbPromise = openDB<ResumeDB>('ai-resume-db', 1, {
			upgrade(db: IDBPDatabase<ResumeDB>) {
				const store = db.createObjectStore('resumes', {
					keyPath: 'id',
					autoIncrement: true,
				});
				store.createIndex('by-username', 'username');
			},
		});
	}
	return dbPromise;
}

export async function addResume(record: Omit<ResumeRecord, 'id' | 'createdAt'>) {
	const db = await initDB();
	const createdAt = Date.now();
	const id = await db.add('resumes', { ...record, createdAt });
	return id;
}

export async function getResume(id: number) {
	const db = await initDB();
	return db.get('resumes', id);
}

export async function listResumesByUser(username?: string) {
	const db = await initDB();
	if (username) {
		return db.getAllFromIndex('resumes', 'by-username', username);
	}
	return db.getAll('resumes');
}

export async function deleteResume(id: number) {
	const db = await initDB();
	await db.delete('resumes', id);
}

export async function clearAllResumes() {
	const db = await initDB();
	const tx = db.transaction('resumes', 'readwrite');
	await tx.objectStore('resumes').clear();
	await tx.done;
}

