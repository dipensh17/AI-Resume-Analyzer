import type { Route } from "./+types/home";
import Navbar from "~/components/Navbar";
import ResumeCard from "~/components/ResumeCard";
import {usePuterStore} from "~/lib/puter";
import {Link, useNavigate} from "react-router";
import {useEffect, useState, useMemo} from "react";
import { listResumesByUser } from "~/lib/db";

export function meta({}: Route.MetaArgs) {
  return [
    { title: "Resumind" },
    { name: "description", content: "Smart feedback for your dream job!" },
  ];
}

export default function Home() {
  const { auth, kv } = usePuterStore();
  const { loginRequired } = usePuterStore();
  const navigate = useNavigate();
  const [resumes, setResumes] = useState<Resume[]>([]);
  const [loadingResumes, setLoadingResumes] = useState(false);
  // DB (IndexedDB) records and search state
  const [dbRecords, setDbRecords] = useState<{
    id?: number;
    username: string;
    filename?: string;
    file?: Blob | null;
    jobTitle: string;
    score: number;
    createdAt: number;
  }[]>([]);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    if (!auth.isAuthenticated && !loginRequired) navigate('/auth?next=/');
  }, [auth.isAuthenticated, loginRequired]);

  useEffect(() => {
    const loadResumes = async () => {
      setLoadingResumes(true);

      const resumes = (await kv.list('resume:*', true)) as KVItem[];

      const parsedResumes = resumes?.map((resume) => (
          JSON.parse(resume.value) as Resume
      ))

      setResumes(parsedResumes || []);
      setLoadingResumes(false);
    }

    loadResumes()
  }, []);

  // Load records from local IndexedDB for history table
  useEffect(() => {
    let mounted = true;
    const loadDB = async () => {
      try {
        const records = await listResumesByUser();
        if (!mounted) return;
        setDbRecords(records || []);
      } catch (err) {
        console.warn('Failed to load local DB records', err);
      }
    };
    loadDB();
    return () => { mounted = false };
  }, []);

  const filteredDbRecords = useMemo(() => {
    const q = (searchTerm || '').toLowerCase();
    const filtered = q
      ? dbRecords.filter(r => (r.jobTitle || '').toLowerCase().includes(q))
      : dbRecords.slice();
    // sort descending by score (highest first)
    filtered.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
    return filtered;
  }, [dbRecords, searchTerm]);

  // Create object URLs for file blobs to be used as links
  const [fileUrls, setFileUrls] = useState<Record<number | string, string>>({});
  useEffect(() => {
    const map: Record<number | string, string> = {};
    filteredDbRecords.forEach((r) => {
      if (r.file && r.id != null) {
        try {
          map[r.id] = URL.createObjectURL(r.file as Blob);
        } catch (e) {
          // ignore
        }
      }
    });
    setFileUrls(map);
    return () => {
      Object.values(map).forEach(url => URL.revokeObjectURL(url));
    }
  }, [filteredDbRecords]);

  return <main className="bg-[url('/images/bg-main.svg')] bg-cover">
    <Navbar />

    <section className="main-section">
      <div className="page-heading py-16">
        <h1>Track Your Applications & Resume Ratings</h1>
        {!loadingResumes && resumes?.length === 0 ? (
            <h2>No resumes found. Upload your first resume to get feedback.</h2>
        ): (
          <h2>Review your submissions and check AI-powered feedback.</h2>
        )}
      </div>
      {loadingResumes && (
          <div className="flex flex-col items-center justify-center">
            <img src="/images/resume-scan-2.gif" className="w-[200px]" />
          </div>
      )}

      {!loadingResumes && resumes.length > 0 && (
        <div className="resumes-section">
          {resumes.map((resume) => (
              <ResumeCard key={resume.id} resume={resume} />
          ))}
        </div>
      )}

      {!loadingResumes && resumes?.length === 0 && (
          <div className="flex flex-col items-center justify-center mt-10 gap-4">
            <Link to="/upload" className="primary-button w-fit text-xl font-semibold">
              Upload Resume
            </Link>
          </div>
      )}

      {/* Local DB History Section */}
      <section className="w-full max-w-[1200px] mx-auto mt-12 p-6 bg-white rounded-2xl shadow-md">
        <div className="flex flex-col gap-4">
          <h2 className="text-2xl font-semibold">Resume History</h2>
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="w-full sm:w-1/2">
              <label className="sr-only" htmlFor="resume-search">Search by job title</label>
              <input
                id="resume-search"
                type="text"
                placeholder="Search job title..."
                className="w-full p-3 border border-gray-200 rounded-lg"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            <div className="text-sm text-gray-500">Showing {filteredDbRecords.length} of {dbRecords.length} records</div>
          </div>

          <div className="overflow-x-auto">
            <table className="min-w-full table-auto">
              <thead>
                <tr className="text-left bg-gray-50">
                  <th className="px-4 py-2">Username</th>
                  <th className="px-4 py-2">Document</th>
                  <th className="px-4 py-2">Job Title</th>
                  <th className="px-4 py-2">Score</th>
                </tr>
              </thead>
              <tbody>
                {filteredDbRecords.map((r) => (
                  <tr key={r.id ?? `${r.username}-${r.createdAt}`} className="border-t">
                    <td className="px-4 py-3 align-top">{r.username}</td>
                    <td className="px-4 py-3 align-top">
                      {r.file ? (
                        <a className="text-blue-600 hover:underline" href={fileUrls[r.id as number] || '#'} target="_blank" rel="noreferrer" download={r.filename}>
                          {r.filename || 'Download'}
                        </a>
                      ) : (
                        r.filename || '—'
                      )}
                    </td>
                    <td className="px-4 py-3 align-top">{r.jobTitle}</td>
                    <td className="px-4 py-3 align-top"><span className="font-semibold">{r.score}</span></td>
                  </tr>
                ))}
                {filteredDbRecords.length === 0 && (
                  <tr>
                    <td colSpan={4} className="px-4 py-6 text-center text-gray-500">No records found.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </section>
  </main>
}
